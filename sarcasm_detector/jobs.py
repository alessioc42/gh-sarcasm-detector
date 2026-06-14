from __future__ import annotations

import logging

from .audio_util import encode_audio_base64
from .config import Config
from .db import Database, JobRecord
from .import_raw import ensure_jobs_for_new_models, sync_models
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

AUDIO_USER_PROMPT = (
    "Listen to the attached audio clip and evaluate whether it is sarcastic."
)


def _build_user_message(
    assets: dict, modality: str, language: str
) -> tuple[str, bytes | None, str | None, str | None, str | None]:
    if modality == "text":
        key = f"transcript_{language}"
        row = assets.get(key)
        if row is None or row["content_text"] is None:
            raise ValueError(f"Missing text asset {key}")
        text = str(row["content_text"])
        context = assets.get("context_de")
        if context and context["content_text"]:
            text = f"Context:\n{context['content_text']}\n\nTranscript:\n{text}"
        return text, None, None, None, None

    key = f"audio_{language}"
    row = assets.get(key)
    if row is None or row["content_blob"] is None:
        raise ValueError(f"Missing audio asset {key}")
    return (
        AUDIO_USER_PROMPT,
        bytes(row["content_blob"]),
        str(row["original_filename"]),
        row["mime_type"],
        row["blob_encoding"],
    )


def _sync_model_capabilities(
    db: Database, conn, client: OllamaClient, model_names: list[str]
) -> None:
    for name in model_names:
        supports, caps = client.model_supports_audio(name)
        model_id = db.upsert_model(
            conn, name, supports_audio=supports, capabilities=caps
        )
        if not supports:
            skipped = db.skip_audio_jobs_for_model(
                conn,
                model_id,
                reason="Model does not report audio capability",
            )
            if skipped:
                logger.info(
                    "Skipped %d audio jobs for model %s (no audio support)",
                    skipped,
                    name,
                )


def _execute_job(
    db: Database,
    conn,
    client: OllamaClient,
    job: JobRecord,
    system_prompt: str,
) -> None:
    assets = db.get_clip_assets(conn, job.clip_id)
    try:
        user_message, audio_bytes, audio_filename, mime_type, blob_encoding = (
            _build_user_message(assets, job.modality, job.language)
        )
    except ValueError as exc:
        db.finish_job(conn, job.id, "failed", last_error=str(exc))
        db.insert_job_output(
            conn,
            job_id=job.id,
            raw_response_body=None,
            request_payload=None,
            http_status=None,
            duration_ms=0,
            error_message=str(exc),
        )
        return

    audio_b64 = None
    if audio_bytes is not None and audio_filename is not None:
        audio_b64 = encode_audio_base64(
            audio_bytes,
            mime_type=str(mime_type) if mime_type else None,
            blob_encoding=str(blob_encoding) if blob_encoding else None,
            original_filename=audio_filename,
        )

    result = client.chat(
        model=job.model_name,
        system_prompt=system_prompt,
        user_message=user_message,
        audio_b64=audio_b64,
    )

    db.insert_job_output(
        conn,
        job_id=job.id,
        raw_response_body=result.raw_body or None,
        request_payload=result.request_payload,
        http_status=result.http_status,
        duration_ms=result.duration_ms,
        error_message=result.error_message,
    )

    if result.error_message:
        db.finish_job(conn, job.id, "failed", last_error=result.error_message)
        logger.error("Job %d failed: %s", job.id, result.error_message)
    else:
        db.finish_job(conn, job.id, "completed")
        logger.info(
            "Job %d completed (%s/%s/%s) in %dms",
            job.id,
            job.model_name,
            job.modality,
            job.language,
            result.duration_ms,
        )


def run_jobs(config: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = Database(config.sqlite_db)
    db.initialize()

    model_names = config.load_models()
    system_prompt = config.load_system_prompt()
    client = OllamaClient(config.ollama_endpoint, config.ollama_api_token)

    try:
        with db.session() as conn:
            model_ids = sync_models(db, conn, model_names)
            ensure_jobs_for_new_models(db, conn, model_ids)
            reset = db.reset_running_jobs(conn)
            if reset:
                logger.info("Reset %d stuck running jobs to pending", reset)
            _sync_model_capabilities(db, conn, client, model_names)

        processed = 0
        while True:
            with db.session() as conn:
                job = db.claim_next_job(conn)
                if job is None:
                    break
                _execute_job(db, conn, client, job, system_prompt)
                processed += 1

        logger.info("Run complete: processed %d jobs", processed)
    finally:
        client.close()


def run_status(config: Config) -> None:
    db = Database(config.sqlite_db)
    db.initialize()

    with db.session() as conn:
        clips = db.count_clips(conn)
        counts = db.job_status_counts(conn)

    print(f"Database: {config.sqlite_db}")
    print(f"Clips: {clips}")
    if counts:
        print("Jobs:")
        for status in sorted(counts):
            print(f"  {status}: {counts[status]}")
    else:
        print("Jobs: none (run import first)")
