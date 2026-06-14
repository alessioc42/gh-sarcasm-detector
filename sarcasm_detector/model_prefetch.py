from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class ModelPrefetcher:
    """Background Ollama model pulls with at most one active download."""

    def __init__(self, client: OllamaClient) -> None:
        self._client = client
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
                self._futures[model_name] = self._executor.submit(
                    self._pull_and_store, model_name
                )
            future = self._futures[model_name]

        future.result()

    def cancel_all(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _pull_and_store(self, model_name: str) -> None:
        if self._client.model_is_available(model_name):
            logger.info("Model %s already available locally", model_name)
            return
        self._client.pull_model(model_name)
