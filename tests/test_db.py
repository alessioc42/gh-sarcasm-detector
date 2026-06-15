from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from sarcasm_detector.db import Database, JobRecord, SCHEMA_VERSION


class TestDatabase:
    def test_initialize_creates_schema(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()
            assert version is not None
            assert int(version["version"]) == SCHEMA_VERSION

    def test_migration_adds_blob_encoding(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (1);
            CREATE TABLE series (id INTEGER PRIMARY KEY, name TEXT UNIQUE);
            CREATE TABLE clips (
                id INTEGER PRIMARY KEY,
                series_id INTEGER,
                source_key TEXT UNIQUE,
                source_archive TEXT,
                source_path TEXT,
                episode TEXT,
                time_start TEXT,
                time_end TEXT,
                ground_truth_sarcasm INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE clip_assets (
                id INTEGER PRIMARY KEY,
                clip_id INTEGER,
                asset_type TEXT,
                mime_type TEXT,
                content_text TEXT,
                content_blob BLOB,
                original_filename TEXT,
                UNIQUE (clip_id, asset_type)
            );
            CREATE TABLE models (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                supports_audio INTEGER,
                capabilities_json TEXT,
                last_checked_at TEXT
            );
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                clip_id INTEGER,
                model_id INTEGER,
                modality TEXT,
                language TEXT,
                status TEXT DEFAULT 'pending',
                attempt_count INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                started_at TEXT,
                finished_at TEXT,
                UNIQUE (clip_id, model_id, modality, language)
            );
            CREATE TABLE job_outputs (
                id INTEGER PRIMARY KEY,
                job_id INTEGER,
                raw_response_body TEXT,
                request_payload_json TEXT,
                http_status INTEGER,
                duration_ms INTEGER,
                error_message TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        conn.commit()
        conn.close()

        db = Database(db_path)
        db.initialize()
        with db.session() as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(clip_assets)").fetchall()
            }
            assert "blob_encoding" in cols
            assert int(conn.execute("SELECT version FROM schema_version").fetchone()[0]) == SCHEMA_VERSION

    def test_migration_adds_job_verdicts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "v2.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (2);
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                clip_id INTEGER,
                model_id INTEGER,
                modality TEXT,
                language TEXT,
                status TEXT DEFAULT 'pending',
                attempt_count INTEGER DEFAULT 0,
                last_error TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                started_at TEXT,
                finished_at TEXT
            );
            """
        )
        conn.commit()
        conn.close()

        db = Database(db_path)
        db.initialize()
        with db.session() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "job_verdicts" in tables
            assert int(conn.execute("SELECT version FROM schema_version").fetchone()[0]) == SCHEMA_VERSION

    def test_get_or_create_series(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            a = tmp_db.get_or_create_series(conn, "Show")
            b = tmp_db.get_or_create_series(conn, "Show")
        assert a == b

    def test_clip_exists_and_insert(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            assert not tmp_db.clip_exists(conn, "arch:01")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="arch:01",
                source_archive="arch.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=None,
            )
            assert clip_id == 1
            assert tmp_db.clip_exists(conn, "arch:01")

    def test_upsert_asset_updates(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            tmp_db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type="transcript_en",
                mime_type=None,
                content_text="v1",
                content_blob=None,
                original_filename="en.txt",
            )
            tmp_db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type="transcript_en",
                mime_type=None,
                content_text="v2",
                content_blob=None,
                original_filename="en.txt",
            )
            row = conn.execute(
                "SELECT content_text FROM clip_assets WHERE clip_id = ?", (clip_id,)
            ).fetchone()
            assert row["content_text"] == "v2"

    def test_upsert_model_preserves_capabilities(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            mid = tmp_db.upsert_model(
                conn, "m1", supports_audio=True, capabilities=["audio"]
            )
            tmp_db.upsert_model(conn, "m1")
            row = conn.execute(
                "SELECT supports_audio, capabilities_json FROM models WHERE id = ?",
                (mid,),
            ).fetchone()
            assert row["supports_audio"] == 1
            assert "audio" in row["capabilities_json"]

    def test_ensure_jobs_idempotent(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            model_id = tmp_db.upsert_model(conn, "m1")
            first = tmp_db.ensure_jobs_for_clip(
                conn, clip_id, [model_id], {("text", "en"), ("audio", "de")}
            )
            second = tmp_db.ensure_jobs_for_clip(
                conn, clip_id, [model_id], {("text", "en"), ("audio", "de")}
            )
        assert first == 2
        assert second == 0

    def test_claim_and_finish_job(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            assert isinstance(job, JobRecord)
            tmp_db.finish_job(conn, job.id, "completed")
            status = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job.id,)
            ).fetchone()
            assert status["status"] == "completed"

    def test_reset_running_jobs(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            assert job is not None
            reset = tmp_db.reset_running_jobs(conn)
            row = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job.id,)
            ).fetchone()
        assert reset == 1
        assert row["status"] == "pending"

    def test_skip_audio_jobs(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            model_id = tmp_db.upsert_model(conn, "m1")
            tmp_db.ensure_jobs_for_clip(
                conn, clip_id, [model_id], {("text", "en"), ("audio", "en")}
            )
            skipped = tmp_db.skip_audio_jobs_for_model(conn, model_id, "no audio")
            row = conn.execute(
                "SELECT status FROM jobs WHERE modality = 'audio'"
            ).fetchone()
        assert skipped == 1
        assert row["status"] == "skipped"

    def test_insert_job_output_and_counts(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            assert job is not None
            tmp_db.insert_job_output(
                conn,
                job_id=job.id,
                raw_response_body='{"sarcastic": true}',
                request_payload={"model": "test"},
                http_status=200,
                duration_ms=10,
            )
            counts = tmp_db.job_status_counts(conn)
            clips = tmp_db.count_clips(conn)
            models = tmp_db.list_model_ids(conn)
        assert counts["running"] == 1
        assert clips == 1
        assert models == [(1, "test-model")]

    def test_claim_returns_none_when_empty(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            assert tmp_db.claim_next_job(conn) is None

    def test_claim_next_job_for_model(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            tmp_db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type="transcript_en",
                mime_type=None,
                content_text="text",
                content_blob=None,
                original_filename="en.txt",
            )
            model_a = tmp_db.upsert_model(conn, "model-a")
            model_b = tmp_db.upsert_model(conn, "model-b")
            tmp_db.ensure_jobs_for_clip(conn, clip_id, [model_a], {("text", "en")})
            tmp_db.ensure_jobs_for_clip(conn, clip_id, [model_b], {("text", "en")})

        with tmp_db.session() as conn:
            job_a = tmp_db.claim_next_job_for_model(conn, model_a)
        with tmp_db.session() as conn:
            job_b = tmp_db.claim_next_job_for_model(conn, model_b)
        assert job_a is not None
        assert job_b is not None
        assert job_a.model_name == "model-a"
        assert job_b.model_name == "model-b"

    def test_fail_pending_jobs_for_model(self, tmp_db: Database) -> None:
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            model_id = tmp_db.upsert_model(conn, "m")
            tmp_db.ensure_jobs_for_clip(conn, clip_id, [model_id], {("text", "en")})
            failed = tmp_db.fail_pending_jobs_for_model(conn, model_id, "pull failed")
            assert failed == 1

    def test_count_pending_jobs_for_model(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            model_id = tmp_db.upsert_model(conn, "test-model")
            assert tmp_db.count_pending_jobs_for_model(conn, model_id) == 1

    def test_migrate_noop_when_current(self, tmp_db: Database) -> None:
        tmp_db.initialize()
        with tmp_db.session() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == SCHEMA_VERSION

    def test_upsert_job_verdict(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job_id = conn.execute("SELECT id FROM jobs").fetchone()["id"]
            tmp_db.upsert_job_verdict(
                conn,
                job_id=job_id,
                verdict="SARCASTIC",
                sarcastic=True,
                confidence=7,
                parse_error=None,
            )
            tmp_db.upsert_job_verdict(
                conn,
                job_id=job_id,
                verdict="NOT_SARCASTIC",
                sarcastic=False,
                confidence=None,
                parse_error=None,
            )
            row = conn.execute(
                "SELECT verdict, sarcastic, confidence FROM job_verdicts WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            counts = tmp_db.verdict_counts(conn)
        assert row["verdict"] == "NOT_SARCASTIC"
        assert row["sarcastic"] == 0
        assert row["confidence"] is None
        assert counts["NOT_SARCASTIC"] == 1

    def test_get_latest_job_output(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job_id = conn.execute("SELECT id FROM jobs").fetchone()["id"]
            tmp_db.insert_job_output(
                conn,
                job_id=job_id,
                raw_response_body="first",
                request_payload={},
                http_status=200,
                duration_ms=1,
            )
            tmp_db.insert_job_output(
                conn,
                job_id=job_id,
                raw_response_body="second",
                request_payload={},
                http_status=200,
                duration_ms=2,
            )
            output = tmp_db.get_latest_job_output(conn, job_id)
        assert output is not None
        assert output["raw_response_body"] == "second"

    def test_list_jobs_for_parsing(self, tmp_db: Database, seed_clip_with_job) -> None:
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            jobs = tmp_db.list_jobs_for_parsing(conn)
        assert len(jobs) == 1
        assert jobs[0].status == "pending"

    def test_session_rollback_on_error(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "rollback.db")
        db.initialize()
        with pytest.raises(RuntimeError):
            with db.session() as conn:
                db.get_or_create_series(conn, "Show")
                raise RuntimeError("boom")
        with db.session() as conn:
            count = conn.execute("SELECT COUNT(*) FROM series").fetchone()[0]
        assert count == 0
