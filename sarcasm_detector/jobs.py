from __future__ import annotations

import logging

from .audio_util import encode_audio_base64
from .config import Config
from .db import (
    FAILURE_KIND_INFERENCE_ERROR,
    FAILURE_KIND_MISSING_ASSET,
    Database,
    JobRecord,
)
from .import_raw import sync_from_config
from .logging_config import configure_logging
from .model_cache import ModelCacheManager, pending_eval_model_names
from .model_prefetch import ModelPrefetcher
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

AUDIO_USER_PROMPT = (
    "Listen to the attached audio clip and evaluate whether it is sarcastic."
)


def _format_status_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{status}={counts[status]}" for status in sorted(counts))


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
    db: Database, conn, client: OllamaClient, model_name: str, model_id: int
) -> int:
    supports, caps = client.model_supports_audio(model_name)
    db.upsert_model(conn, model_name, supports_audio=supports, capabilities=caps)
    if caps:
        logger.info(
            "Model %s capabilities: %s (audio=%s)",
            model_name,
            ", ".join(caps),
            supports,
        )
    else:
        logger.info("Model %s: no capabilities reported (audio=%s)", model_name, supports)

    skipped = 0
    if not supports:
        skipped = db.skip_audio_jobs_for_model(
            conn,
            model_id,
            reason="Model does not report audio capability",
        )
        if skipped:
            logger.info(
                "Skipped %d audio jobs for %s (no audio support)",
                skipped,
                model_name,
            )
    return skipped


def _handle_inference_failure(
    db: Database,
    conn,
    job: JobRecord,
    error_message: str,
    *,
    max_job_attempts: int,
) -> None:
    if job.attempt_count < max_job_attempts:
        db.requeue_job(conn, job.id, last_error=error_message)
        logger.warning(
            "Job %d failed (attempt %d/%d), requeued: %s",
            job.id,
            job.attempt_count,
            max_job_attempts,
            error_message,
        )
        return

    db.finish_job(
        conn,
        job.id,
        "failed",
        last_error=error_message,
        failure_kind=FAILURE_KIND_INFERENCE_ERROR,
    )
    logger.error(
        "Job %d failed permanently after %d attempts (%s/%s/%s): %s",
        job.id,
        job.attempt_count,
        job.model_name,
        job.modality,
        job.language,
        error_message,
    )


def _execute_job(
    db: Database,
    conn,
    client: OllamaClient,
    job: JobRecord,
    config: Config,
    *,
    progress: str,
) -> None:
    logger.info(
        "%s job %d: clip %d, prompt %s, %s/%s",
        progress,
        job.id,
        job.clip_id,
        job.prompt_slug,
        job.modality,
        job.language,
    )
    assets = db.get_clip_assets(conn, job.clip_id)
    try:
        user_message, audio_bytes, audio_filename, mime_type, blob_encoding = (
            _build_user_message(assets, job.modality, job.language)
        )
    except ValueError as exc:
        db.finish_job(
            conn,
            job.id,
            "failed",
            last_error=str(exc),
            failure_kind=FAILURE_KIND_MISSING_ASSET,
        )
        db.insert_job_output(
            conn,
            job_id=job.id,
            raw_response_body=None,
            request_payload=None,
            http_status=None,
            duration_ms=0,
            error_message=str(exc),
        )
        logger.error("Job %d failed before inference: %s", job.id, exc)
        return

    audio_b64 = None
    if audio_bytes is not None and audio_filename is not None:
        audio_b64 = encode_audio_base64(
            audio_bytes,
            mime_type=str(mime_type) if mime_type else None,
            blob_encoding=str(blob_encoding) if blob_encoding else None,
            original_filename=audio_filename,
        )

    system_prompt = config.read_prompt_text(job.prompt_slug)
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
        _handle_inference_failure(
            db,
            conn,
            job,
            result.error_message,
            max_job_attempts=config.max_job_attempts,
        )
    else:
        db.finish_job(conn, job.id, "completed")
        logger.info(
            "Job %d completed (%s/%s/%s, prompt=%s) in %dms",
            job.id,
            job.model_name,
            job.modality,
            job.language,
            job.prompt_slug,
            result.duration_ms,
        )


def _run_jobs_for_model(
    db: Database,
    client: OllamaClient,
    model_id: int,
    model_name: str,
    config: Config,
    *,
    pending_total: int,
) -> int:
    processed = 0
    while True:
        with db.session() as conn:
            job = db.claim_next_job_for_model(conn, model_id)
            if job is None:
                break
            progress = f"[{model_name} {processed + 1}/{pending_total}]"
            _execute_job(db, conn, client, job, config, progress=progress)
            processed += 1
    return processed


def run_jobs(config: Config) -> None:
    configure_logging()
    db = Database(config.sqlite_db)
    db.initialize()

    model_names = config.load_models()
    if not model_names:
        logger.error("No models listed in %s", config.models_path)
        return

    prompts = config.load_prompts()
    if not prompts:
        logger.error("No .txt prompt files found in %s", config.prompts_dir)
        return

    logger.info("Starting evaluation run")
    logger.info("Database: %s", config.sqlite_db)
    logger.info("Ollama endpoint: %s", config.ollama_endpoint)
    logger.info(
        "Models to evaluate: %d from %s",
        len(model_names),
        config.models_path,
    )
    logger.info(
        "Prompts: %d from %s (max attempts per job: %d)",
        len(prompts),
        config.prompts_dir,
        config.max_job_attempts,
    )

    client = OllamaClient(config.ollama_endpoint, config.ollama_api_token)
    cache = ModelCacheManager(client, config)
    current_model: list[str] = [""]
    remaining_models: set[str] = set(model_names)
    prefetcher: ModelPrefetcher | None = None

    def protected_supplier() -> set[str]:
        protected = set(remaining_models)
        if current_model[0]:
            protected.add(current_model[0])
        if prefetcher is not None:
            protected |= prefetcher.scheduled_models()
        protected |= pending_eval_model_names(db, model_names)
        return protected

    def pending_supplier() -> set[str]:
        return pending_eval_model_names(db, model_names)

    prefetcher = ModelPrefetcher(
        client,
        cache,
        eval_models=model_names,
        protected_supplier=protected_supplier,
        pending_supplier=pending_supplier,
    )

    try:
        with db.session() as conn:
            _, model_ids, synced_prompt_ids, jobs_created = sync_from_config(
                db, conn, config
            )
            clips = db.count_clips(conn)
            counts = db.job_status_counts(conn)
            reset = db.reset_running_jobs(conn)

        pending_total = counts.get("pending", 0)
        logger.info(
            "Synced %d models, %d prompts (%d new jobs created)",
            len(model_ids),
            len(synced_prompt_ids),
            jobs_created,
        )
        logger.info("Dataset: %d clips, %d pending jobs", clips, pending_total)
        if counts:
            logger.info("Current job status: %s", _format_status_counts(counts))
        if reset:
            logger.info("Reset %d stuck running jobs to pending", reset)

        total_processed = 0
        for index, model_name in enumerate(model_names):
            remaining_models = set(model_names[index:])
            current_model[0] = model_name
            model_num = index + 1
            logger.info(
                "--- Model %d/%d: %s ---",
                model_num,
                len(model_names),
                model_name,
            )

            with db.session() as conn:
                model_id = db.upsert_model(conn, model_name)
                pending_before_pull = db.count_pending_jobs_for_model(conn, model_id)

            if pending_before_pull == 0:
                logger.info(
                    "No pending jobs for %s, skipping model",
                    model_name,
                )
                continue

            logger.info(
                "%d pending jobs queued for %s before model pull",
                pending_before_pull,
                model_name,
            )

            try:
                prefetcher.ensure_pulled(model_name)
            except Exception as exc:
                logger.error(
                    "Failed to pull model %s: %s (leaving %d jobs pending for retry)",
                    model_name,
                    exc,
                    pending_before_pull,
                )
                continue

            with db.session() as conn:
                pending = db.count_pending_jobs_for_model(conn, model_id)
                skipped = _sync_model_capabilities(
                    db, conn, client, model_name, model_id
                )
                pending = db.count_pending_jobs_for_model(conn, model_id)

            if skipped:
                logger.info(
                    "Pending jobs for %s after capability check: %d (skipped %d audio)",
                    model_name,
                    pending,
                    skipped,
                )

            if pending == 0:
                logger.info(
                    "No runnable jobs left for %s after capability check, skipping evaluation",
                    model_name,
                )
                cache.mark_done(model_name)
                continue

            if index + 1 < len(model_names):
                next_model = model_names[index + 1]
                logger.info(
                    "Prefetching next model in background: %s",
                    next_model,
                )
                prefetcher.schedule_pull(next_model)

            logger.info("Running %d pending jobs for %s", pending, model_name)
            processed = _run_jobs_for_model(
                db,
                client,
                model_id,
                model_name,
                config,
                pending_total=pending,
            )
            total_processed += processed

            with db.session() as conn:
                model_counts = db.job_status_counts_for_model(conn, model_id)

            logger.info(
                "Finished model %s: processed %d jobs this run (%s)",
                model_name,
                processed,
                _format_status_counts(model_counts),
            )
            cache.mark_done(model_name)

        with db.session() as conn:
            final_counts = db.job_status_counts(conn)

        logger.info(
            "Run complete: processed %d jobs in this session",
            total_processed,
        )
        logger.info("Final job status: %s", _format_status_counts(final_counts))
        cache.log_cache_summary()
    finally:
        logger.info("Shutting down runner")
        prefetcher.cancel_all()
        client.close()


def run_status(config: Config) -> None:
    db = Database(config.sqlite_db)
    db.initialize()

    with db.session() as conn:
        model_names, _, prompt_ids, jobs_created = sync_from_config(db, conn, config)
        clips = db.count_clips(conn)
        counts = db.job_status_counts(conn)
        failure_counts = db.failure_kind_counts(conn)
        pending = counts.get("pending", 0)
        verdict_counts = db.verdict_counts(conn)

    print(f"Database: {config.sqlite_db}")
    print(f"Models: {len(model_names)} (from {config.models_path})")
    print(f"Prompts: {len(prompt_ids)} (from {config.prompts_dir})")
    print(f"Max job attempts: {config.max_job_attempts}")
    print(f"Clips: {clips}")
    if clips == 0:
        print("Jobs: none (run import first)")
        return
    print(f"Jobs to run: {pending}")
    if jobs_created:
        print(f"New jobs created from sync: {jobs_created}")
    if counts:
        print("Jobs:")
        for status in sorted(counts):
            print(f"  {status}: {counts[status]}")
    else:
        print("Jobs: none")
    if failure_counts:
        print("Failure kinds:")
        for kind in sorted(failure_counts):
            print(f"  {kind}: {failure_counts[kind]}")
    if verdict_counts:
        print("Verdicts (completed jobs only):")
        for verdict in sorted(verdict_counts):
            print(f"  {verdict}: {verdict_counts[verdict]}")
