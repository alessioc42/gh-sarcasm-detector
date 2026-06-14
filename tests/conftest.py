from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from sarcasm_detector.config import Config
from sarcasm_detector.db import Database


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture
def config(tmp_path: Path) -> Config:
    (tmp_path / "system_prompt.txt").write_text("You are a test prompt.")
    (tmp_path / "models.txt").write_text("test-model\n# comment\n")
    return Config(
        ollama_endpoint="http://localhost:11434",
        ollama_api_token=None,
        sqlite_db=tmp_path / "test.db",
        system_prompt_path=tmp_path / "system_prompt.txt",
        models_path=tmp_path / "models.txt",
        raw_data_dir=tmp_path / "raw_data",
    )


@pytest.fixture
def config_with_db(config: Config, tmp_db: Database) -> Config:
    return replace(config, sqlite_db=tmp_db.path)


def _write_bilingual_clip(base: Path, *, deadpool_en: bool = False) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "misc.txt").write_text(
        "IS_SARCASM=True\nEPISODE=S01E01\nTIME_START=01:00\nTIME_END=01:30\n"
    )
    en_name = "en.txt.txt" if deadpool_en else "en.txt"
    (base / en_name).write_text('Speaker: "Sure, that sounds great."\n')
    (base / "de.txt").write_text('Speaker: "Klar, das klingt toll."\n')
    (base / "en.audio.mp3").write_bytes(b"\xff\xfb" + b"\x00" * 64)
    (base / "de.audio.mp3").write_bytes(b"\xff\xfb" + b"\x01" * 64)


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    clip = tmp_path / "staging" / "Family Guy" / "01"
    _write_bilingual_clip(clip)
    zip_path = tmp_path / "forschungsmethoden-Family Guy.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in (tmp_path / "staging").rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(tmp_path / "staging"))
    return zip_path


@pytest.fixture
def deadpool_zip(tmp_path: Path) -> Path:
    clip = tmp_path / "staging" / "Deadpool_Clips" / "Clip 2"
    _write_bilingual_clip(clip, deadpool_en=True)
    (clip / "en.audio2.mp3").write_bytes(b"\xff\xfb" + b"\x02" * 64)
    (clip / "en.audio.mp3").unlink(missing_ok=True)
    zip_path = tmp_path / "Deadpool_Clips.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in (tmp_path / "staging").rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(tmp_path / "staging"))
    return zip_path


@pytest.fixture
def werner_tar(tmp_path: Path) -> Path:
    clip = tmp_path / "staging" / "Werner" / "volles-Roaaa" / "01"
    clip.mkdir(parents=True)
    (clip / "DE.md").write_text("**Werner:** Roaaa\n")
    (clip / "SPEC.md").write_text("Kontext: Beispiel\n")
    (clip / "DE.m4a").write_bytes(b"\x00" * 32)
    tar_path = tmp_path / "Werner.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for file in (tmp_path / "staging").rglob("*"):
            if file.is_file():
                tf.add(file, arcname=str(file.relative_to(tmp_path / "staging")))
    return tar_path


@pytest.fixture
def raw_data_dir(tmp_path: Path, sample_zip: Path) -> Path:
    raw = tmp_path / "raw_data"
    raw.mkdir()
    shutil.copy(sample_zip, raw / sample_zip.name)
    return raw


@pytest.fixture
def seed_clip_with_job():
    def _seed(db: Database, *, modality: str = "text", language: str = "en") -> int:
        with db.session() as conn:
            series_id = db.get_or_create_series(conn, "Test")
            clip_id = db.insert_clip(
                conn,
                series_id=series_id,
                source_key="test:01",
                source_archive="test.zip",
                source_path="01",
                episode="S01E01",
                time_start="01:00",
                time_end="01:30",
                ground_truth_sarcasm=True,
            )
            db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type=f"transcript_{language}",
                mime_type=None,
                content_text="Test transcript",
                content_blob=None,
                original_filename=f"{language}.txt",
            )
            model_id = db.upsert_model(conn, "test-model")
            db.ensure_jobs_for_clip(conn, clip_id, [model_id], {(modality, language)})
        return clip_id

    return _seed
