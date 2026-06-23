from __future__ import annotations

import threading
import time
from unittest import mock

import pytest

from sarcasm_detector.model_cache import ModelCacheManager
from sarcasm_detector.model_prefetch import ModelPrefetcher
from sarcasm_detector.ollama_client import OllamaClient


def _prefetcher(client: mock.Mock, cache: mock.Mock | None = None) -> ModelPrefetcher:
    cache = cache or mock.Mock(spec=ModelCacheManager)
    return ModelPrefetcher(
        client,
        cache,
        eval_models=["model-a", "model-b"],
        protected_supplier=lambda: set(),
        pending_supplier=lambda: set(),
    )


class TestModelPrefetcher:
    def test_schedule_pull_idempotent(self) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.pull_model.return_value = None
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)
        prefetcher.schedule_pull("model-a")
        prefetcher.schedule_pull("model-a")
        prefetcher.ensure_pulled("model-a")
        assert client.pull_model.call_count == 1
        cache.ensure_space_for_pull.assert_called_once()
        prefetcher.cancel_all()

    def test_ensure_pulled_skips_when_available(self) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = True
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)
        prefetcher.ensure_pulled("model-a")
        client.pull_model.assert_not_called()
        cache.ensure_space_for_pull.assert_not_called()
        prefetcher.cancel_all()

    def test_ensure_pulled_waits_for_scheduled(self) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)
        prefetcher.schedule_pull("model-b")
        prefetcher.ensure_pulled("model-b")
        client.pull_model.assert_called_once_with("model-b")
        prefetcher.cancel_all()

    def test_ensure_pulled_propagates_error(self) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.pull_model.side_effect = RuntimeError("network down")
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)
        with pytest.raises(RuntimeError, match="network down"):
            prefetcher.ensure_pulled("model-a")
        prefetcher.cancel_all()

    def test_scheduled_models(self) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)
        prefetcher.schedule_pull("model-b")
        assert prefetcher.scheduled_models() == {"model-b"}
        prefetcher.cancel_all()

    def test_prefetch_next_before_current_blocks_on_wrong_model(self) -> None:
        pull_order: list[str] = []
        pull_started = threading.Event()
        release_pull = threading.Event()

        def pull_model(model_name: str) -> None:
            pull_order.append(model_name)
            pull_started.set()
            release_pull.wait(timeout=2)

        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.pull_model.side_effect = pull_model
        cache = mock.Mock(spec=ModelCacheManager)
        prefetcher = _prefetcher(client, cache)

        prefetcher.schedule_pull("model-b")
        waiter = threading.Thread(target=prefetcher.ensure_pulled, args=("model-a",))
        waiter.start()
        assert pull_started.wait(timeout=2)
        assert pull_order == ["model-b"]

        release_pull.set()
        waiter.join(timeout=2)
        assert pull_order == ["model-b", "model-a"]
        prefetcher.cancel_all()
