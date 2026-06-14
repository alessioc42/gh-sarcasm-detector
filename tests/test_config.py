from __future__ import annotations

import os
from pathlib import Path

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
        assert cfg.raw_data_dir == Path("raw_data")
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
