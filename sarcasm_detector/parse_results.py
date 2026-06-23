from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .config import Config
from .db import Database

logger = logging.getLogger(__name__)

VERDICT_SARCASTIC = "SARCASTIC"
VERDICT_NOT_SARCASTIC = "NOT_SARCASTIC"
VERDICT_LLM_ERR = "LLM_ERR"

_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedFields:
    sarcastic: bool
    confidence: int | None


@dataclass(frozen=True)
class ParsedVerdict:
    verdict: str
    sarcastic: bool | None
    confidence: int | None
    parse_error: str | None


def parse_confidence(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 10:
        return value
    if isinstance(value, float):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.lstrip("-").isdigit():
            return None
        parsed = int(stripped)
        if 0 <= parsed <= 10:
            return parsed
    return None


def parse_sarcastic(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes"}:
            return True
        if lower in {"false", "0", "no"}:
            return False
    return None


def extract_assistant_text(raw_response_body: str | None) -> str:
    if not raw_response_body:
        return ""
    try:
        envelope = json.loads(raw_response_body)
    except json.JSONDecodeError:
        return raw_response_body

    if isinstance(envelope, dict):
        message = envelope.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    return raw_response_body


def _strip_markdown_fences(text: str) -> str:
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_schema_object(obj: object) -> ParsedFields | None:
    if not isinstance(obj, dict) or "sarcastic" not in obj:
        return None
    sarcastic = parse_sarcastic(obj["sarcastic"])
    if sarcastic is None:
        return None
    confidence = (
        parse_confidence(obj["confidence"]) if "confidence" in obj else None
    )
    return ParsedFields(sarcastic=sarcastic, confidence=confidence)


def _try_parse_json_object(text: str) -> ParsedFields | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _parse_schema_object(obj)


def _find_schema_json_in_text(text: str) -> ParsedFields | None:
    stripped = _strip_markdown_fences(text)
    parsed = _try_parse_json_object(stripped)
    if parsed is not None:
        return parsed

    start = stripped.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(stripped)):
            char = stripped[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : index + 1]
                    parsed = _try_parse_json_object(candidate)
                    if parsed is not None:
                        return parsed
                    break
        start = stripped.find("{", start + 1)
    return None


def find_schema_json(text: str) -> ParsedFields | None:
    return _find_schema_json_in_text(text)


def classify_completed_response(*, raw_body: str | None) -> ParsedVerdict:
    assistant_text = extract_assistant_text(raw_body)
    fields = find_schema_json(assistant_text)
    if fields is None:
        return ParsedVerdict(
            verdict=VERDICT_LLM_ERR,
            sarcastic=None,
            confidence=None,
            parse_error="no JSON object with sarcastic key",
        )

    verdict = VERDICT_SARCASTIC if fields.sarcastic else VERDICT_NOT_SARCASTIC
    return ParsedVerdict(
        verdict=verdict,
        sarcastic=fields.sarcastic,
        confidence=fields.confidence,
        parse_error=None,
    )


def run_parse(config: Config) -> None:
    db = Database(config.sqlite_db)
    db.initialize()

    skipped_unparsed = 0
    counts: dict[str, int] = {}

    with db.session() as conn:
        jobs = db.list_jobs_for_parsing(conn)

    for job in jobs:
        if job.status != "completed":
            with db.session() as conn:
                db.delete_job_verdict(conn, job.id)
            skipped_unparsed += 1
            continue

        with db.session() as conn:
            output = db.get_latest_job_output(conn, job.id)
            raw_body = (
                str(output["raw_response_body"])
                if output and output["raw_response_body"]
                else None
            )
            parsed = classify_completed_response(raw_body=raw_body)
            db.upsert_job_verdict(
                conn,
                job_id=job.id,
                verdict=parsed.verdict,
                sarcastic=parsed.sarcastic,
                confidence=parsed.confidence,
                parse_error=parsed.parse_error,
            )

        counts[parsed.verdict] = counts.get(parsed.verdict, 0) + 1

    if skipped_unparsed:
        logger.info(
            "Skipped %d non-completed jobs (no verdict written)",
            skipped_unparsed,
        )

    logger.info(
        "Parse complete: %s",
        ", ".join(f"{verdict}={counts[verdict]}" for verdict in sorted(counts)),
    )
