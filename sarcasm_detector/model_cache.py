from __future__ import annotations

import logging
import shutil

from .config import Config
from .ollama_client import InstalledModel, OllamaClient

logger = logging.getLogger(__name__)


class ModelCacheManager:
    """Disk-aware Ollama model cache with tiered eviction."""

    def __init__(self, client: OllamaClient, config: Config) -> None:
        self._client = client
        self._config = config
        self._done_models: set[str] = set()

    def disk_free_bytes(self) -> int:
        path = self._config.ollama_models_dir
        if not path.exists():
            path = path.parent
        if not path.exists():
            path = path.parent
        return shutil.disk_usage(path).free

    def installed_models(self) -> dict[str, InstalledModel]:
        return {m.name: m for m in self._client.list_installed_models()}

    def mark_done(self, model_name: str) -> None:
        self._done_models.add(model_name)

    def bytes_needed_for_pull(self, target: str) -> int:
        if self._client.model_is_available(target):
            return 0
        installed = self.installed_models()
        entry = installed.get(target)
        if entry is not None and entry.size_bytes > 0:
            payload = entry.size_bytes
        else:
            payload = self._config.model_pull_reserve_bytes
        return payload + self._config.min_free_disk_bytes

    def ensure_space_for_pull(
        self,
        target: str,
        *,
        protected: set[str],
        eval_models: set[str],
        pending_eval_models: set[str],
    ) -> None:
        if self._client.model_is_available(target):
            return

        needed = self.bytes_needed_for_pull(target)
        if needed == 0:
            return

        while self.disk_free_bytes() < needed:
            candidate = self._pick_eviction_candidate(
                protected=protected,
                eval_models=eval_models,
                pending_eval_models=pending_eval_models,
            )
            if candidate is None:
                free = self.disk_free_bytes()
                raise OSError(
                    f"Insufficient disk space for {target}: "
                    f"need {needed} bytes free, have {free} bytes and no evictable models"
                )
            logger.info(
                "Evicting model %s (%d bytes) to free disk for %s",
                candidate.name,
                candidate.size_bytes,
                target,
            )
            self._client.delete_model(candidate.name)

        free = self.disk_free_bytes()
        logger.info(
            "Disk space OK for pull of %s (%d bytes free, need %d)",
            target,
            free,
            needed,
        )

    def _pick_eviction_candidate(
        self,
        *,
        protected: set[str],
        eval_models: set[str],
        pending_eval_models: set[str],
    ) -> InstalledModel | None:
        installed = list(self.installed_models().values())
        evictable = [
            m
            for m in installed
            if m.name not in protected and m.name not in pending_eval_models
        ]
        if not evictable:
            return None

        orphans = [m for m in evictable if m.name not in eval_models]
        if orphans:
            return self._oldest(orphans)

        completed_eval = [
            m
            for m in evictable
            if m.name in eval_models and m.name in self._done_models
        ]
        if completed_eval:
            return self._oldest(completed_eval)

        return None

    @staticmethod
    def _oldest(models: list[InstalledModel]) -> InstalledModel:
        return min(
            models,
            key=lambda m: m.modified_at or "",
        )

    def log_cache_summary(self) -> None:
        installed = self.installed_models()
        free = self.disk_free_bytes()
        logger.info(
            "Model cache: %d installed, %d bytes free on disk",
            len(installed),
            free,
        )
        if installed:
            names = ", ".join(sorted(installed))
            logger.info("Installed models retained: %s", names)


def pending_eval_model_names(
    db,
    model_names: list[str],
) -> set[str]:
    pending: set[str] = set()
    with db.session() as conn:
        for model_id, name in db.list_model_ids(conn):
            if name in model_names and db.count_pending_jobs_for_model(conn, model_id) > 0:
                pending.add(name)
    return pending
