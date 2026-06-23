from __future__ import annotations

import os
from pathlib import Path

import pytest

from sarcasm_detector.config import Config


class TestConfig:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SQLITE_DB", raising=False)
        monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)
        monkeypatch.delenv("MAX_JOB_ATTEMPTS", raising=False)
        monkeypatch.chdir(tmp_path)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "default.txt").write_text("prompt")
        (tmp_path / "models.txt").write_text("model-a\n")
        cfg = Config.from_env()
        assert cfg.ollama_endpoint == "http://localhost:11434"
        assert cfg.sqlite_db == Path("sarcasm.db")
        assert cfg.raw_data_dir == Path("raw_data")
        assert cfg.prompts_dir == Path("prompts")
        assert cfg.max_job_attempts == 3
        prompts = cfg.load_prompts()
        assert len(prompts) == 1
        assert prompts[0][0] == "default"
        assert prompts[0][1].name == "default.txt"
        assert cfg.read_prompt_text("default") == "prompt"
        assert cfg.load_models() == ["model-a"]

    def test_from_env_custom(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "custom_prompts"
        prompts_dir.mkdir()
        (prompts_dir / "v1.txt").write_text("sys")
        monkeypatch.setenv("OLLAMA_ENDPOINT", "http://ollama:11434/")
        monkeypatch.setenv("OLLAMA_API_TOKEN", "secret")
        monkeypatch.setenv("SQLITE_DB", str(tmp_path / "custom.db"))
        monkeypatch.setenv("PROMPTS_DIR", str(prompts_dir))
        monkeypatch.setenv("MODELS_PATH", str(tmp_path / "m.txt"))
        monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "raw"))
        monkeypatch.setenv("MAX_JOB_ATTEMPTS", "5")
        (tmp_path / "m.txt").write_text("# skip\nmodel-b\n")
        cfg = Config.from_env()
        assert cfg.ollama_endpoint == "http://ollama:11434"
        assert cfg.ollama_api_token == "secret"
        assert cfg.max_job_attempts == 5
        assert cfg.load_models() == ["model-b"]
        assert cfg.load_prompts() == [("v1", prompts_dir / "v1.txt")]

    def test_from_env_disk_settings(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        models_dir = tmp_path / "models_store"
        monkeypatch.setenv("OLLAMA_MODELS_DIR", str(models_dir))
        monkeypatch.setenv("MIN_FREE_DISK_BYTES", "1000")
        monkeypatch.setenv("MODEL_PULL_RESERVE_BYTES", "2000")
        cfg = Config.from_env()
        assert cfg.ollama_models_dir == models_dir
        assert cfg.min_free_disk_bytes == 1000
        assert cfg.model_pull_reserve_bytes == 2000

    def test_empty_token_becomes_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_API_TOKEN", "")
        cfg = Config.from_env()
        assert cfg.ollama_api_token is None

    def test_invalid_max_job_attempts_defaults_to_three(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAX_JOB_ATTEMPTS", "not-a-number")
        cfg = Config.from_env()
        assert cfg.max_job_attempts == 3

    def test_missing_models_file(self, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "default.txt").write_text("prompt")
        cfg = Config(
            ollama_endpoint="http://localhost:11434",
            ollama_api_token=None,
            sqlite_db=tmp_path / "db",
            prompts_dir=prompts_dir,
            models_path=tmp_path / "missing-models.txt",
            raw_data_dir=tmp_path / "raw",
            max_job_attempts=3,
            ollama_models_dir=tmp_path / "ollama_models",
            min_free_disk_bytes=2_000_000_000,
            model_pull_reserve_bytes=8_000_000_000,
        )
        assert cfg.load_models() == []

    def test_read_prompt_text_missing_raises(self, tmp_path: Path) -> None:
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        cfg = Config(
            ollama_endpoint="http://localhost:11434",
            ollama_api_token=None,
            sqlite_db=tmp_path / "db",
            prompts_dir=prompts_dir,
            models_path=tmp_path / "models.txt",
            raw_data_dir=tmp_path / "raw",
            max_job_attempts=3,
            ollama_models_dir=tmp_path / "ollama_models",
            min_free_disk_bytes=2_000_000_000,
            model_pull_reserve_bytes=8_000_000_000,
        )
        with pytest.raises(KeyError, match="Prompt not found"):
            cfg.read_prompt_text("missing")
