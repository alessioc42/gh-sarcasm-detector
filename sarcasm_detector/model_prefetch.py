from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from .model_cache import ModelCacheManager
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ModelPrefetcher:
    """Background Ollama model pulls with at most one active download."""

    def __init__(
        self,
        client: OllamaClient,
        cache: ModelCacheManager,
        *,
        eval_models: list[str],
        protected_supplier: Callable[[], set[str]],
        pending_supplier: Callable[[], set[str]],
    ) -> None:
        self._client = client
        self._cache = cache
        self._eval_models = set(eval_models)
        self._protected_supplier = protected_supplier
        self._pending_supplier = pending_supplier
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama-pull")
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    def schedule_pull(self, model_name: str) -> None:
        with self._lock:
            if model_name in self._futures:
                return
            logger.info("Scheduling background pull for %s", model_name)
            self._futures[model_name] = self._executor.submit(
                self._pull_and_store, model_name
            )

    def ensure_pulled(self, model_name: str) -> None:
        with self._lock:
            if model_name not in self._futures:
                logger.info("Starting on-demand pull for %s", model_name)
                self._futures[model_name] = self._executor.submit(
                    self._pull_and_store, model_name
                )
            else:
                logger.info("Waiting for scheduled pull of %s to finish", model_name)
            future = self._futures[model_name]

        future.result()
        logger.info("Model %s is ready for evaluation", model_name)

    def scheduled_models(self) -> set[str]:
        with self._lock:
            return set(self._futures.keys())

    def cancel_all(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _pull_and_store(self, model_name: str) -> None:
        if self._client.model_is_available(model_name):
            logger.info("Model %s already available locally", model_name)
            return

        protected = self._protected_supplier() | {model_name}
        pending = self._pending_supplier()
        self._cache.ensure_space_for_pull(
            model_name,
            protected=protected,
            eval_models=self._eval_models,
            pending_eval_models=pending,
        )

        logger.info("Downloading model %s from Ollama", model_name)
        self._client.pull_model(model_name)
        logger.info("Download complete for %s", model_name)
