from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

from sarcasm_detector.db import Database
from sarcasm_detector.jobs import (
    AUDIO_USER_PROMPT,
    _build_user_message,
    _execute_job,
    _sync_model_capabilities,
    run_jobs,
    run_status,
)
from sarcasm_detector.ollama_client import ChatResult, OllamaClient


def _make_assets_row(**kwargs) -> sqlite3.Row:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cols = list(kwargs.keys())
    conn.execute(f"CREATE TABLE t ({', '.join(f'{c} TEXT' for c in cols)})")
    conn.execute(
        f"INSERT INTO t ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        tuple(str(v) if v is not None else None for v in kwargs.values()),
    )
    return conn.execute("SELECT * FROM t").fetchone()


class TestBuildUserMessage:
    def test_text_transcript(self) -> None:
        assets = {
            "transcript_en": _make_assets_row(content_text="Hello"),
        }
        msg, audio, fname, mime, enc = _build_user_message(assets, "text", "en")
        assert msg == "Hello"
        assert audio is None

    def test_text_with_context(self) -> None:
        assets = {
            "transcript_de": _make_assets_row(content_text="Hallo"),
            "context_de": _make_assets_row(content_text="Kontext"),
        }
        msg, *_ = _build_user_message(assets, "text", "de")
        assert "Kontext" in msg
        assert "Hallo" in msg

    def test_audio_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing audio"):
            _build_user_message({}, "audio", "en")

    def test_audio_asset(self) -> None:
        assets = {
            "audio_en": _make_assets_row(
                content_blob=b"audio",
                original_filename="en.audio.mp3",
                mime_type="audio/mpeg",
                blob_encoding="raw",
            ),
        }
        # sqlite3 stores blob as bytes when using proper insert - fix test
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE a (content_blob BLOB, original_filename TEXT, mime_type TEXT, blob_encoding TEXT)"
        )
        conn.execute(
            "INSERT INTO a VALUES (?, ?, ?, ?)",
            (b"audio-bytes", "en.audio.mp3", "audio/mpeg", "raw"),
        )
        row = conn.execute("SELECT * FROM a").fetchone()
        msg, audio, fname, mime, enc = _build_user_message({"audio_en": row}, "audio", "en")
        assert msg == AUDIO_USER_PROMPT
        assert audio == b"audio-bytes"
        assert fname == "en.audio.mp3"


class TestSyncModelCapabilities:
    def test_skips_audio_jobs_for_text_only_model(self, tmp_db: Database) -> None:
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
                asset_type="audio_en",
                mime_type="audio/mpeg",
                content_text=None,
                content_blob=b"x",
                original_filename="en.mp3",
            )
            model_id = tmp_db.upsert_model(conn, "text-only")
            tmp_db.ensure_jobs_for_clip(
                conn, clip_id, [model_id], {("text", "en"), ("audio", "en")}
            )

        client = mock.Mock(spec=OllamaClient)
        client.model_supports_audio.return_value = (False, ["completion"])

        with tmp_db.session() as conn:
            _sync_model_capabilities(tmp_db, conn, client, ["text-only"])
            audio_status = conn.execute(
                "SELECT status FROM jobs WHERE modality = 'audio'"
            ).fetchone()
        assert audio_status["status"] == "skipped"


class TestExecuteJob:
    def test_execute_text_job_success(
        self, tmp_db: Database, seed_clip_with_job
    ) -> None:
        seed_clip_with_job(tmp_db)
        client = mock.Mock(spec=OllamaClient)
        client.chat.return_value = ChatResult(
            raw_body='{"sarcastic": true}',
            http_status=200,
            duration_ms=50,
            request_payload={"model": "test-model"},
        )
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            assert job is not None
            _execute_job(tmp_db, conn, client, job, "system")
            status = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job.id,)
            ).fetchone()
            output = conn.execute(
                "SELECT raw_response_body FROM job_outputs WHERE job_id = ?",
                (job.id,),
            ).fetchone()
        assert status["status"] == "completed"
        assert "sarcastic" in output["raw_response_body"]

    def test_execute_job_missing_asset_fails(self, tmp_db: Database) -> None:
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

        client = mock.Mock(spec=OllamaClient)
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            _execute_job(tmp_db, conn, client, job, "system")
            status = conn.execute(
                "SELECT status, last_error FROM jobs WHERE id = ?", (job.id,)
            ).fetchone()
        assert status["status"] == "failed"
        assert "Missing text" in status["last_error"]
        client.chat.assert_not_called()

    def test_execute_job_chat_failure(
        self, tmp_db: Database, seed_clip_with_job
    ) -> None:
        seed_clip_with_job(tmp_db)
        client = mock.Mock(spec=OllamaClient)
        client.chat.return_value = ChatResult(
            raw_body="error",
            http_status=500,
            duration_ms=5,
            request_payload={},
            error_message="HTTP 500",
        )
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            _execute_job(tmp_db, conn, client, job, "system")
            status = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job.id,)
            ).fetchone()
        assert status["status"] == "failed"

    @mock.patch("sarcasm_detector.jobs.encode_audio_base64", return_value="YmFzZTY0")
    def test_execute_audio_job(self, mock_encode: mock.Mock, tmp_db: Database) -> None:
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
                asset_type="audio_en",
                mime_type="audio/mpeg",
                content_text=None,
                content_blob=b"audio",
                original_filename="en.mp3",
            )
            model_id = tmp_db.upsert_model(conn, "m")
            tmp_db.ensure_jobs_for_clip(conn, clip_id, [model_id], {("audio", "en")})

        client = mock.Mock(spec=OllamaClient)
        client.chat.return_value = ChatResult(
            raw_body="ok",
            http_status=200,
            duration_ms=10,
            request_payload={},
        )
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            _execute_job(tmp_db, conn, client, job, "system")
        mock_encode.assert_called_once()
        client.chat.assert_called_once()
        assert client.chat.call_args.kwargs["audio_b64"] == "YmFzZTY0"


class TestRunJobs:
    @mock.patch("sarcasm_detector.jobs.OllamaClient")
    def test_run_jobs_processes_queue(
        self,
        mock_client_cls: mock.Mock,
        config_with_db: Config,
        tmp_db: Database,
        seed_clip_with_job,
    ) -> None:
        config = config_with_db
        seed_clip_with_job(tmp_db)
        instance = mock_client_cls.return_value
        instance.model_supports_audio.return_value = (False, [])
        instance.chat.return_value = ChatResult(
            raw_body="ok",
            http_status=200,
            duration_ms=1,
            request_payload={},
        )
        run_jobs(config)
        instance.close.assert_called_once()

    @mock.patch("sarcasm_detector.jobs.OllamaClient")
    def test_run_jobs_resets_stuck(
        self,
        mock_client_cls: mock.Mock,
        config_with_db: Config,
        tmp_db: Database,
        seed_clip_with_job,
    ) -> None:
        config = config_with_db
        seed_clip_with_job(tmp_db)
        with tmp_db.session() as conn:
            job = tmp_db.claim_next_job(conn)
            assert job is not None

        instance = mock_client_cls.return_value
        instance.model_supports_audio.return_value = (False, [])
        instance.chat.return_value = ChatResult(
            raw_body="ok",
            http_status=200,
            duration_ms=1,
            request_payload={},
        )
        run_jobs(config)
        with tmp_db.session() as conn:
            status = conn.execute("SELECT status FROM jobs").fetchone()
        assert status["status"] == "completed"


class TestRunStatus:
    def test_run_status_empty(self, config, capsys: pytest.CaptureFixture[str]) -> None:
        config.sqlite_db.parent.mkdir(parents=True, exist_ok=True)
        run_status(config)
        out = capsys.readouterr().out
        assert "Clips: 0" in out
        assert "none (run import first)" in out

    def test_run_status_with_jobs(
        self, config_with_db: Config, tmp_db: Database, seed_clip_with_job, capsys
    ) -> None:
        config = config_with_db
        seed_clip_with_job(tmp_db)
        run_status(config)
        out = capsys.readouterr().out
        assert "Clips: 1" in out
        assert "pending" in out
