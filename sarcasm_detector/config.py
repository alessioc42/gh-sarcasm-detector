from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIN_FREE_DISK_BYTES = 2_000_000_000
DEFAULT_MODEL_PULL_RESERVE_BYTES = 8_000_000_000


def _default_ollama_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS_DIR") or os.environ.get("OLLAMA_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".ollama" / "models"


def _parse_positive_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    ollama_endpoint: str
    ollama_api_token: str | None
    sqlite_db: Path
    prompts_dir: Path
    models_path: Path
    raw_data_dir: Path
    max_job_attempts: int
    ollama_models_dir: Path
    min_free_disk_bytes: int
    model_pull_reserve_bytes: int

    @classmethod
    def from_env(cls) -> Config:
        token = os.environ.get("OLLAMA_API_TOKEN") or None
        if token == "":
            token = None

        max_attempts_raw = os.environ.get("MAX_JOB_ATTEMPTS", "3")
        try:
            max_job_attempts = max(1, int(max_attempts_raw))
        except ValueError:
            max_job_attempts = 3

        return cls(
            ollama_endpoint=os.environ.get(
                "OLLAMA_ENDPOINT", "http://localhost:11434"
            ).rstrip("/"),
            ollama_api_token=token,
            sqlite_db=Path(os.environ.get("SQLITE_DB", "sarcasm.db")),
            prompts_dir=Path(os.environ.get("PROMPTS_DIR", "prompts")),
            models_path=Path(os.environ.get("MODELS_PATH", "models.txt")),
            raw_data_dir=Path(os.environ.get("RAW_DATA_DIR", "raw_data")),
            max_job_attempts=max_job_attempts,
            ollama_models_dir=_default_ollama_models_dir(),
            min_free_disk_bytes=_parse_positive_int(
                os.environ.get("MIN_FREE_DISK_BYTES"),
                DEFAULT_MIN_FREE_DISK_BYTES,
            ),
            model_pull_reserve_bytes=_parse_positive_int(
                os.environ.get("MODEL_PULL_RESERVE_BYTES"),
                DEFAULT_MODEL_PULL_RESERVE_BYTES,
            ),
        )

    def load_prompts(self) -> list[tuple[str, Path]]:
        if not self.prompts_dir.is_dir():
            return []
        prompts: list[tuple[str, Path]] = []
        for path in sorted(self.prompts_dir.glob("*.txt")):
            prompts.append((path.stem, path))
        return prompts

    def read_prompt_text(self, slug: str) -> str:
        for prompt_slug, path in self.load_prompts():
            if prompt_slug == slug:
                return path.read_text(encoding="utf-8").strip()
        raise KeyError(f"Prompt not found: {slug}")

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
