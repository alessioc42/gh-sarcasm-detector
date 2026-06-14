from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from sarcasm_detector.config import Config


class TestConfig:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SQLITE_DB", raising=False)
        monkeypatch.delenv("OLLAMA_ENDPOINT", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "system_prompt.txt").write_text("prompt")
        (tmp_path / "models.txt").write_text("model-a\n")
        cfg = Config.from_env()
        assert cfg.ollama_endpoint == "http://localhost:11434"
        assert cfg.sqlite_db == Path("sarcasm.db")
        assert cfg.load_system_prompt() == "prompt"
        assert cfg.load_models() == ["model-a"]

    def test_from_env_custom(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("OLLAMA_ENDPOINT", "http://ollama:11434/")
        monkeypatch.setenv("OLLAMA_API_TOKEN", "secret")
        monkeypatch.setenv("SQLITE_DB", str(tmp_path / "custom.db"))
        monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(tmp_path / "sp.txt"))
        monkeypatch.setenv("MODELS_PATH", str(tmp_path / "m.txt"))
        monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "raw"))
        (tmp_path / "sp.txt").write_text("sys")
        (tmp_path / "m.txt").write_text("# skip\nmodel-b\n")
        cfg = Config.from_env()
        assert cfg.ollama_endpoint == "http://ollama:11434"
        assert cfg.ollama_api_token == "secret"
        assert cfg.load_models() == ["model-b"]

    def test_empty_token_becomes_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_API_TOKEN", "")
        cfg = Config.from_env()
        assert cfg.ollama_api_token is None

    def test_missing_models_file(self, tmp_path: Path) -> None:
        cfg = Config(
            ollama_endpoint="http://localhost:11434",
            ollama_api_token=None,
            sqlite_db=tmp_path / "db",
            system_prompt_path=tmp_path / "missing.txt",
            models_path=tmp_path / "missing-models.txt",
            raw_data_dir=tmp_path / "raw",
        )
        assert cfg.load_models() == []

    def test_docker_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("sarcasm_detector.config._in_docker", lambda: True)
        monkeypatch.setenv("MODELS_PATH", str(tmp_path / "models.txt"))
        (tmp_path / "models.txt").write_text("docker-model\n")
        cfg = Config.from_env()
        assert cfg.sqlite_db == Path("/data/db/sarcasm.db")
        assert cfg.raw_data_dir == Path("/data/raw_data")
        assert cfg.load_models() == ["docker-model"]

    def test_docker_config_file_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("sarcasm_detector.config._in_docker", lambda: True)
        config_dir = Path("/data/config")
        monkeypatch.setattr(
            Path,
            "is_file",
            lambda self: str(self) == str(config_dir / "system_prompt.txt"),
        )
        cfg = Config.from_env()
        assert cfg.system_prompt_path == config_dir / "system_prompt.txt"

    def test_in_docker(self) -> None:
        from sarcasm_detector.config import _in_docker

        assert isinstance(_in_docker(), bool)
