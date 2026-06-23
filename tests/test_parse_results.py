from __future__ import annotations

import json

import pytest

from sarcasm_detector.db import Database
from sarcasm_detector.parse_results import (
    VERDICT_LLM_ERR,
    VERDICT_NOT_SARCASTIC,
    VERDICT_SARCASTIC,
    classify_completed_response,
    extract_assistant_text,
    find_schema_json,
    parse_confidence,
    parse_sarcastic,
    run_parse,
)


class TestParseConfidence:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (8, 8),
            (0, 0),
            (10, 10),
            ("7", 7),
            ("10", 10),
            (None, None),
            (0.9, None),
            (7.5, None),
            (11, None),
            (-1, None),
            (True, None),
            ("high", None),
            ("8.0", None),
        ],
    )
    def test_parse_confidence(self, value: object, expected: int | None) -> None:
        assert parse_confidence(value) == expected


class TestParseSarcastic:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("true", True),
            ("false", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            (1, None),
            ("maybe", None),
        ],
    )
    def test_parse_sarcastic(self, value: object, expected: bool | None) -> None:
        assert parse_sarcastic(value) == expected


class TestExtractAssistantText:
    def test_ollama_envelope(self) -> None:
        body = json.dumps(
            {"message": {"content": '{"sarcastic": true, "confidence": 5}'}}
        )
        assert extract_assistant_text(body) == '{"sarcastic": true, "confidence": 5}'

    def test_plain_text(self) -> None:
        assert extract_assistant_text('{"sarcastic": false}') == '{"sarcastic": false}'

    def test_empty(self) -> None:
        assert extract_assistant_text(None) == ""
        assert extract_assistant_text("") == ""


class TestFindSchemaJson:
    def test_valid_json_with_confidence(self) -> None:
        fields = find_schema_json('{"sarcastic": true, "confidence": 8}')
        assert fields is not None
        assert fields.sarcastic is True
        assert fields.confidence == 8

    def test_float_confidence_becomes_null(self) -> None:
        fields = find_schema_json('{"sarcastic": false, "confidence": 0.9}')
        assert fields is not None
        assert fields.sarcastic is False
        assert fields.confidence is None

    def test_missing_confidence(self) -> None:
        fields = find_schema_json('{"sarcastic": true}')
        assert fields is not None
        assert fields.confidence is None

    def test_markdown_fence(self) -> None:
        text = '```json\n{"sarcastic": true, "confidence": 3}\n```'
        fields = find_schema_json(text)
        assert fields is not None
        assert fields.sarcastic is True
        assert fields.confidence == 3

    def test_prose_with_trailing_json(self) -> None:
        text = 'Here is my answer:\n{"sarcastic": false, "confidence": 2}'
        fields = find_schema_json(text)
        assert fields is not None
        assert fields.sarcastic is False
        assert fields.confidence == 2

    def test_invalid_returns_none(self) -> None:
        assert find_schema_json("just prose") is None
        assert find_schema_json('{"confidence": 5}') is None
        assert find_schema_json('{"sarcastic": "maybe"}') is None


class TestClassifyCompletedResponse:
    def test_sarcastic(self) -> None:
        body = json.dumps({"message": {"content": '{"sarcastic": true, "confidence": 9}'}})
        result = classify_completed_response(raw_body=body)
        assert result.verdict == VERDICT_SARCASTIC
        assert result.sarcastic is True
        assert result.confidence == 9

    def test_not_sarcastic(self) -> None:
        result = classify_completed_response(
            raw_body='{"sarcastic": false, "confidence": 1}'
        )
        assert result.verdict == VERDICT_NOT_SARCASTIC

    def test_llm_err(self) -> None:
        result = classify_completed_response(raw_body="Sorry, I cannot answer that.")
        assert result.verdict == VERDICT_LLM_ERR
        assert result.parse_error == "no JSON object with sarcastic key"


class TestRunParse:
    def _complete_job(
        self,
        db: Database,
        *,
        raw_body: str,
        status: str = "completed",
        last_error: str | None = None,
    ) -> int:
        with db.session() as conn:
            series_id = db.get_or_create_series(conn, "Show")
            clip_id = db.insert_clip(
                conn,
                series_id=series_id,
                source_key=f"k-{raw_body[:8]}",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            model_id = db.upsert_model(conn, "m")
            prompt_id = db.upsert_prompt(conn, "default", "default.txt")
            db.ensure_jobs_for_clip(
                conn, clip_id, [model_id], [prompt_id], {("text", "en")}
            )
            job_id = conn.execute("SELECT id FROM jobs").fetchone()["id"]
            if status != "pending":
                db.finish_job(conn, job_id, status, last_error=last_error)
            if status == "completed":
                db.insert_job_output(
                    conn,
                    job_id=job_id,
                    raw_response_body=raw_body,
                    request_payload={},
                    http_status=200,
                    duration_ms=1,
                )
        return int(job_id)

    def test_run_parse_integration(
        self, config_with_db, tmp_db: Database
    ) -> None:
        body = json.dumps(
            {"message": {"content": '{"sarcastic": true, "confidence": 6}'}}
        )
        self._complete_job(tmp_db, raw_body=body)
        run_parse(config_with_db)

        with tmp_db.session() as conn:
            row = conn.execute(
                "SELECT verdict, confidence FROM job_verdicts"
            ).fetchone()
            counts = tmp_db.verdict_counts(conn)

        assert row["verdict"] == VERDICT_SARCASTIC
        assert row["confidence"] == 6
        assert counts[VERDICT_SARCASTIC] == 1

    def test_run_parse_idempotent(self, config_with_db, tmp_db: Database) -> None:
        self._complete_job(tmp_db, raw_body='{"sarcastic": false, "confidence": 2}')
        run_parse(config_with_db)
        run_parse(config_with_db)

        with tmp_db.session() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_verdicts").fetchone()[0]
        assert count == 1

    def test_run_parse_skips_non_completed(
        self, config_with_db, tmp_db: Database, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._complete_job(tmp_db, raw_body='{"sarcastic": true}', status="pending")
        run_parse(config_with_db)

        with tmp_db.session() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_verdicts").fetchone()[0]
        assert count == 0
        assert "Skipped 1 non-completed jobs" in caplog.text

    def test_run_parse_failed_job_no_verdict(
        self, config_with_db, tmp_db: Database
    ) -> None:
        self._complete_job(
            tmp_db,
            raw_body="",
            status="failed",
            last_error="pull failed",
        )
        run_parse(config_with_db)

        with tmp_db.session() as conn:
            count = conn.execute("SELECT COUNT(*) FROM job_verdicts").fetchone()[0]
        assert count == 0
