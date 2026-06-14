from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ollama_endpoint: str
    ollama_api_token: str | None
    sqlite_db: Path
    system_prompt_path: Path
    models_path: Path
    raw_data_dir: Path

    @classmethod
    def from_env(cls) -> Config:
        token = os.environ.get("OLLAMA_API_TOKEN") or None
        if token == "":
            token = None

        return cls(
            ollama_endpoint=os.environ.get(
                "OLLAMA_ENDPOINT", "http://localhost:11434"
            ).rstrip("/"),
            ollama_api_token=token,
            sqlite_db=Path(os.environ.get("SQLITE_DB", "sarcasm.db")),
            system_prompt_path=Path(
                os.environ.get("SYSTEM_PROMPT_PATH", "system_prompt.txt")
            ),
            models_path=Path(os.environ.get("MODELS_PATH", "models.txt")),
            raw_data_dir=Path(os.environ.get("RAW_DATA_DIR", "raw_data")),
        )

    def load_system_prompt(self) -> str:
        return self.system_prompt_path.read_text(encoding="utf-8").strip()

    def load_models(self) -> list[str]:
        if not self.models_path.is_file():
            return []
        models: list[str] = []
        for line in self.models_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)
        return models
