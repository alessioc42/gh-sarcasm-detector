from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _in_docker() -> bool:
    return Path("/.dockerenv").exists()


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
        docker = _in_docker()
        default_db = Path("/data/db/sarcasm.db") if docker else Path("sarcasm.db")
        default_raw = Path("/data/raw_data") if docker else Path("raw_data")

        def _resolve_path(env_key: str, docker_default: str, local_default: str) -> Path:
            if env_key in os.environ:
                return Path(os.environ[env_key])
            if docker:
                override = Path("/data/config") / Path(docker_default).name
                if override.is_file():
                    return override
                return Path(docker_default)
            return Path(local_default)

        system_prompt = _resolve_path(
            "SYSTEM_PROMPT_PATH", "/app/system_prompt.txt", "system_prompt.txt"
        )
        models_path = _resolve_path("MODELS_PATH", "/app/models.txt", "models.txt")

        token = os.environ.get("OLLAMA_API_TOKEN") or None
        if token == "":
            token = None

        return cls(
            ollama_endpoint=os.environ.get(
                "OLLAMA_ENDPOINT", "http://localhost:11434"
            ).rstrip("/"),
            ollama_api_token=token,
            sqlite_db=Path(os.environ.get("SQLITE_DB", str(default_db))),
            system_prompt_path=system_prompt,
            models_path=models_path,
            raw_data_dir=Path(os.environ.get("RAW_DATA_DIR", str(default_raw))),
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
