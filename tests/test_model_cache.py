from __future__ import annotations

import logging
from unittest import mock

import pytest

from sarcasm_detector.config import Config
from sarcasm_detector.model_cache import ModelCacheManager
from sarcasm_detector.ollama_client import InstalledModel, OllamaClient


def _config(tmp_path) -> Config:
    return Config(
        ollama_endpoint="http://localhost:11434",
        ollama_api_token=None,
        sqlite_db=tmp_path / "test.db",
        prompts_dir=tmp_path / "prompts",
        models_path=tmp_path / "models.txt",
        raw_data_dir=tmp_path / "raw",
        max_job_attempts=3,
        ollama_models_dir=tmp_path / "ollama_models",
        min_free_disk_bytes=100,
        model_pull_reserve_bytes=500,
    )


class TestModelCacheManager:
    def test_bytes_needed_when_installed(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = True
        cache = ModelCacheManager(client, _config(tmp_path))
        assert cache.bytes_needed_for_pull("m1") == 0

    def test_bytes_needed_unknown_size_uses_reserve(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = []
        cache = ModelCacheManager(client, _config(tmp_path))
        assert cache.bytes_needed_for_pull("new-model") == 600

    def test_bytes_needed_known_size(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = [
            InstalledModel(name="partial", size_bytes=250, modified_at=None),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        assert cache.bytes_needed_for_pull("partial") == 350

    def test_ensure_space_skips_when_installed(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = True
        cache = ModelCacheManager(client, _config(tmp_path))
        cache.ensure_space_for_pull(
            "m1",
            protected=set(),
            eval_models={"m1"},
            pending_eval_models=set(),
        )
        client.delete_model.assert_not_called()

    def test_ensure_space_no_eviction_when_enough_disk(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = []
        cache = ModelCacheManager(client, _config(tmp_path))
        with mock.patch.object(cache, "disk_free_bytes", return_value=10_000):
            cache.ensure_space_for_pull(
                "new",
                protected=set(),
                eval_models={"new"},
                pending_eval_models=set(),
            )
        client.delete_model.assert_not_called()

    def test_evicts_orphan_first(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = [
            InstalledModel(name="orphan", size_bytes=200, modified_at="2020-01-01"),
            InstalledModel(name="eval-done", size_bytes=200, modified_at="2021-01-01"),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        cache.mark_done("eval-done")
        with mock.patch.object(cache, "disk_free_bytes", side_effect=[50, 10_000, 10_000]):
            cache.ensure_space_for_pull(
                "new",
                protected=set(),
                eval_models={"eval-done", "new"},
                pending_eval_models=set(),
            )
        client.delete_model.assert_called_once_with("orphan")

    def test_evicts_completed_eval_when_no_orphans(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = [
            InstalledModel(name="old-done", size_bytes=200, modified_at="2020-01-01"),
            InstalledModel(name="new-done", size_bytes=200, modified_at="2021-01-01"),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        cache.mark_done("old-done")
        cache.mark_done("new-done")
        with mock.patch.object(cache, "disk_free_bytes", side_effect=[50, 10_000, 10_000]):
            cache.ensure_space_for_pull(
                "target",
                protected=set(),
                eval_models={"old-done", "new-done", "target"},
                pending_eval_models=set(),
            )
        client.delete_model.assert_called_once_with("old-done")

    def test_never_evicts_protected(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = [
            InstalledModel(name="orphan", size_bytes=200, modified_at=None),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        with mock.patch.object(cache, "disk_free_bytes", return_value=50):
            with pytest.raises(OSError, match="Insufficient disk space"):
                cache.ensure_space_for_pull(
                    "new",
                    protected={"orphan"},
                    eval_models={"new"},
                    pending_eval_models=set(),
                )
        client.delete_model.assert_not_called()

    def test_never_evicts_pending_eval(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.model_is_available.return_value = False
        client.list_installed_models.return_value = [
            InstalledModel(name="pending-model", size_bytes=200, modified_at=None),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        with mock.patch.object(cache, "disk_free_bytes", return_value=50):
            with pytest.raises(OSError, match="Insufficient disk space"):
                cache.ensure_space_for_pull(
                    "new",
                    protected=set(),
                    eval_models={"pending-model", "new"},
                    pending_eval_models={"pending-model"},
                )
        client.delete_model.assert_not_called()

    def test_log_cache_summary(self, tmp_path, caplog: pytest.LogCaptureFixture) -> None:
        client = mock.Mock(spec=OllamaClient)
        client.list_installed_models.return_value = [
            InstalledModel(name="m1", size_bytes=100, modified_at=None),
        ]
        cache = ModelCacheManager(client, _config(tmp_path))
        with caplog.at_level(logging.INFO, logger="sarcasm_detector.model_cache"):
            with mock.patch.object(cache, "disk_free_bytes", return_value=999):
                cache.log_cache_summary()
        assert "Model cache: 1 installed" in caplog.text
        assert "m1" in caplog.text

    def test_disk_free_bytes_uses_parent_when_missing(self, tmp_path) -> None:
        client = mock.Mock(spec=OllamaClient)
        models_dir = tmp_path / "missing" / "models"
        cfg = _config(tmp_path)
        cfg = Config(
            ollama_endpoint=cfg.ollama_endpoint,
            ollama_api_token=cfg.ollama_api_token,
            sqlite_db=cfg.sqlite_db,
            prompts_dir=cfg.prompts_dir,
            models_path=cfg.models_path,
            raw_data_dir=cfg.raw_data_dir,
            max_job_attempts=cfg.max_job_attempts,
            ollama_models_dir=models_dir,
            min_free_disk_bytes=cfg.min_free_disk_bytes,
            model_pull_reserve_bytes=cfg.model_pull_reserve_bytes,
        )
        tmp_path.mkdir(exist_ok=True)
        cache = ModelCacheManager(client, cfg)
        assert cache.disk_free_bytes() > 0
