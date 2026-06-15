from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL REFERENCES series(id),
    source_key TEXT NOT NULL UNIQUE,
    source_archive TEXT NOT NULL,
    source_path TEXT NOT NULL,
    episode TEXT,
    time_start TEXT,
    time_end TEXT,
    ground_truth_sarcasm INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS clip_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    mime_type TEXT,
    content_text TEXT,
    content_blob BLOB,
    blob_encoding TEXT NOT NULL DEFAULT 'raw',
    original_filename TEXT NOT NULL,
    UNIQUE (clip_id, asset_type)
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    supports_audio INTEGER,
    capabilities_json TEXT,
    last_checked_at TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'audio')),
    language TEXT NOT NULL CHECK (language IN ('en', 'de')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (clip_id, model_id, modality, language)
);

CREATE TABLE IF NOT EXISTS job_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    raw_response_body TEXT,
    request_payload_json TEXT,
    http_status INTEGER,
    duration_ms INTEGER,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_clips_source_key ON clips(source_key);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class JobRecord:
    id: int
    clip_id: int
    model_id: int
    model_name: str
    modality: str
    language: str
    attempt_count: int


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.session() as conn:
            conn.executescript(SCHEMA_SQL)
            row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            else:
                self._migrate(conn, int(row["version"]))

    def _migrate(self, conn: sqlite3.Connection, version: int) -> None:
        if version >= SCHEMA_VERSION:
            return

        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(clip_assets)").fetchall()
        }
        if "blob_encoding" not in columns:
            conn.execute(
                """
                ALTER TABLE clip_assets
                ADD COLUMN blob_encoding TEXT NOT NULL DEFAULT 'raw'
                """
            )

        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def get_or_create_series(self, conn: sqlite3.Connection, name: str) -> int:
        conn.execute("INSERT OR IGNORE INTO series (name) VALUES (?)", (name,))
        row = conn.execute("SELECT id FROM series WHERE name = ?", (name,)).fetchone()
        assert row is not None
        return int(row["id"])

    def clip_exists(self, conn: sqlite3.Connection, source_key: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM clips WHERE source_key = ?", (source_key,)
        ).fetchone()
        return row is not None

    def insert_clip(
        self,
        conn: sqlite3.Connection,
        *,
        series_id: int,
        source_key: str,
        source_archive: str,
        source_path: str,
        episode: str | None,
        time_start: str | None,
        time_end: str | None,
        ground_truth_sarcasm: bool | None,
    ) -> int:
        gt = None if ground_truth_sarcasm is None else int(ground_truth_sarcasm)
        cur = conn.execute(
            """
            INSERT INTO clips (
                series_id, source_key, source_archive, source_path,
                episode, time_start, time_end, ground_truth_sarcasm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                series_id,
                source_key,
                source_archive,
                source_path,
                episode,
                time_start,
                time_end,
                gt,
            ),
        )
        return int(cur.lastrowid)

    def upsert_asset(
        self,
        conn: sqlite3.Connection,
        *,
        clip_id: int,
        asset_type: str,
        mime_type: str | None,
        content_text: str | None,
        content_blob: bytes | None,
        blob_encoding: str = "raw",
        original_filename: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO clip_assets (
                clip_id, asset_type, mime_type, content_text, content_blob,
                blob_encoding, original_filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id, asset_type) DO UPDATE SET
                mime_type = excluded.mime_type,
                content_text = excluded.content_text,
                content_blob = excluded.content_blob,
                blob_encoding = excluded.blob_encoding,
                original_filename = excluded.original_filename
            """,
            (
                clip_id,
                asset_type,
                mime_type,
                content_text,
                content_blob,
                blob_encoding,
                original_filename,
            ),
        )

    def upsert_model(
        self,
        conn: sqlite3.Connection,
        name: str,
        *,
        supports_audio: bool | None = None,
        capabilities: list[str] | None = None,
    ) -> int:
        caps_json = json.dumps(capabilities) if capabilities is not None else None
        sa = None if supports_audio is None else int(supports_audio)
        conn.execute(
            """
            INSERT INTO models (name, supports_audio, capabilities_json, last_checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                supports_audio = COALESCE(excluded.supports_audio, models.supports_audio),
                capabilities_json = COALESCE(excluded.capabilities_json, models.capabilities_json),
                last_checked_at = COALESCE(excluded.last_checked_at, models.last_checked_at)
            """,
            (name, sa, caps_json, utc_now() if capabilities is not None else None),
        )
        row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
        assert row is not None
        return int(row["id"])

    def ensure_jobs_for_clip(
        self,
        conn: sqlite3.Connection,
        clip_id: int,
        model_ids: Iterable[int],
        available: set[tuple[str, str]],
    ) -> int:
        created = 0
        for model_id in model_ids:
            for modality, language in available:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs (clip_id, model_id, modality, language)
                    VALUES (?, ?, ?, ?)
                    """,
                    (clip_id, model_id, modality, language),
                )
                if cur.rowcount:
                    created += 1
        return created

    def reset_running_jobs(self, conn: sqlite3.Connection) -> int:
        cur = conn.execute(
            """
            UPDATE jobs
            SET status = 'pending', started_at = NULL
            WHERE status = 'running'
            """
        )
        return cur.rowcount

    def claim_next_job(self, conn: sqlite3.Connection) -> JobRecord | None:
        return self._claim_next_job(conn, model_id=None)

    def claim_next_job_for_model(
        self, conn: sqlite3.Connection, model_id: int
    ) -> JobRecord | None:
        return self._claim_next_job(conn, model_id=model_id)

    def count_pending_jobs_for_model(
        self, conn: sqlite3.Connection, model_id: int
    ) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM jobs
            WHERE model_id = ? AND status = 'pending'
            """,
            (model_id,),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def fail_pending_jobs_for_model(
        self,
        conn: sqlite3.Connection,
        model_id: int,
        reason: str,
    ) -> int:
        now = utc_now()
        cur = conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', finished_at = ?, last_error = ?
            WHERE model_id = ? AND status = 'pending'
            """,
            (now, reason, model_id),
        )
        return cur.rowcount

    def _claim_next_job(
        self, conn: sqlite3.Connection, *, model_id: int | None
    ) -> JobRecord | None:
        conn.execute("BEGIN IMMEDIATE")
        if model_id is None:
            row = conn.execute(
                """
                SELECT j.id, j.clip_id, j.model_id, m.name AS model_name,
                       j.modality, j.language, j.attempt_count
                FROM jobs j
                JOIN models m ON m.id = j.model_id
                WHERE j.status = 'pending'
                ORDER BY j.id
                LIMIT 1
                """
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT j.id, j.clip_id, j.model_id, m.name AS model_name,
                       j.modality, j.language, j.attempt_count
                FROM jobs j
                JOIN models m ON m.id = j.model_id
                WHERE j.status = 'pending' AND j.model_id = ?
                ORDER BY j.id
                LIMIT 1
                """,
                (model_id,),
            ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        now = utc_now()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'running', started_at = ?, attempt_count = attempt_count + 1
            WHERE id = ?
            """,
            (now, row["id"]),
        )
        conn.execute("COMMIT")
        return JobRecord(
            id=int(row["id"]),
            clip_id=int(row["clip_id"]),
            model_id=int(row["model_id"]),
            model_name=str(row["model_name"]),
            modality=str(row["modality"]),
            language=str(row["language"]),
            attempt_count=int(row["attempt_count"]) + 1,
        )

    def finish_job(
        self,
        conn: sqlite3.Connection,
        job_id: int,
        status: str,
        *,
        last_error: str | None = None,
    ) -> None:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, finished_at = ?, last_error = ?
            WHERE id = ?
            """,
            (status, utc_now(), last_error, job_id),
        )

    def skip_audio_jobs_for_model(
        self, conn: sqlite3.Connection, model_id: int, reason: str
    ) -> int:
        cur = conn.execute(
            """
            UPDATE jobs
            SET status = 'skipped', finished_at = ?, last_error = ?
            WHERE model_id = ? AND modality = 'audio' AND status = 'pending'
            """,
            (utc_now(), reason, model_id),
        )
        return cur.rowcount

    def insert_job_output(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: int,
        raw_response_body: str | None,
        request_payload: dict[str, Any] | None,
        http_status: int | None,
        duration_ms: int | None,
        error_message: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO job_outputs (
                job_id, raw_response_body, request_payload_json,
                http_status, duration_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                raw_response_body,
                json.dumps(request_payload) if request_payload else None,
                http_status,
                duration_ms,
                error_message,
            ),
        )

    def get_clip_assets(
        self, conn: sqlite3.Connection, clip_id: int
    ) -> dict[str, sqlite3.Row]:
        rows = conn.execute(
            "SELECT * FROM clip_assets WHERE clip_id = ?", (clip_id,)
        ).fetchall()
        return {str(r["asset_type"]): r for r in rows}

    def job_status_counts(self, conn: sqlite3.Connection) -> dict[str, int]:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status"
        ).fetchall()
        return {str(r["status"]): int(r["cnt"]) for r in rows}

    def job_status_counts_for_model(
        self, conn: sqlite3.Connection, model_id: int
    ) -> dict[str, int]:
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS cnt FROM jobs
            WHERE model_id = ?
            GROUP BY status
            """,
            (model_id,),
        ).fetchall()
        return {str(r["status"]): int(r["cnt"]) for r in rows}

    def list_model_ids(self, conn: sqlite3.Connection) -> list[tuple[int, str]]:
        rows = conn.execute("SELECT id, name FROM models ORDER BY id").fetchall()
        return [(int(r["id"]), str(r["name"])) for r in rows]

    def count_clips(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM clips").fetchone()
        return int(row["cnt"]) if row else 0
