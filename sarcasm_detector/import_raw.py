from __future__ import annotations

import logging
import re
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .audio_util import mime_for_filename
from .config import Config
from .db import Database
from .logging_config import configure_logging

logger = logging.getLogger(__name__)

ARCHIVE_EXTENSIONS = {".zip", ".tar.gz", ".tgz", ".tar"}

SERIES_OVERRIDES = {
    "Deadpool_Clips.zip": "Deadpool",
    "fackjughoete.zip": "Fack ju Göhte",
    "forschungsmethoden-Family Guy.zip": "Family Guy",
    "Forschungsmethoden_SouthPark.zip": "South Park",
    "young-sheldon.zip": "Young Sheldon",
    "Werner.tar.gz": "Werner",
}

TEXT_ASSET_MAP = {
    "en.txt": "transcript_en",
    "en.txt.txt": "transcript_en",
    "de.txt": "transcript_de",
    "DE.md": "transcript_de",
    "SPEC.md": "context_de",
}

AUDIO_PATTERNS = [
    (re.compile(r"^en\.audio\d*\.(mp3|wav|m4a)$", re.I), "audio_en"),
    (re.compile(r"^de\.audio\d*\.(mp3|wav|m4a)$", re.I), "audio_de"),
    (re.compile(r"^DE\.(m4a|mp3|wav)$", re.I), "audio_de"),
]


@dataclass
class ClipMetadata:
    episode: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    ground_truth_sarcasm: bool | None = None


@dataclass
class PendingAsset:
    asset_type: str
    path: Path
    mime_type: str | None = None
    content_text: str | None = None
    content_blob: bytes | None = None


@dataclass
class PendingClip:
    relative_path: str
    series_name: str
    metadata: ClipMetadata = field(default_factory=ClipMetadata)
    assets: list[PendingAsset] = field(default_factory=list)


def parse_misc_txt(text: str) -> ClipMetadata:
    meta = ClipMetadata()
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().upper()
        value = value.strip()
        if key == "IS_SARCASM":
            meta.ground_truth_sarcasm = value.lower() in {"true", "1", "yes"}
        elif key in {"EPISODE", "FILM"}:
            meta.episode = value
        elif key == "TIME_START":
            meta.time_start = value
        elif key == "TIME_END":
            meta.time_end = value
    return meta


def is_clip_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {p.name for p in path.iterdir()}
    if "misc.txt" in names:
        return True
    if "DE.md" in names:
        return True
    return False


def find_clip_directories(root: Path) -> list[Path]:
    clips: list[Path] = []
    for path in sorted(root.rglob("*")):
        if "aup3" in path.parts:
            continue
        if is_clip_directory(path):
            clips.append(path)
    return clips


def read_series_name(extract_root: Path, archive_name: str) -> str:
    if archive_name in SERIES_OVERRIDES:
        override = SERIES_OVERRIDES[archive_name]
        if override:
            return override

    for candidate in extract_root.rglob("series.txt"):
        text = candidate.read_text(encoding="utf-8").strip()
        if text:
            return text

    stem = archive_name
    for ext in (".tar.gz", ".zip", ".tgz", ".tar"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem.replace("_", " ").replace("-", " ").strip()


def classify_file(filename: str) -> str | None:
    lower = filename.lower()
    for pattern, asset_type in AUDIO_PATTERNS:
        if pattern.match(filename) or pattern.match(lower):
            return asset_type

    for name, asset_type in TEXT_ASSET_MAP.items():
        if filename == name or lower == name.lower():
            return asset_type
    return None


def collect_clip(
    clip_dir: Path, extract_root: Path, series_name: str
) -> PendingClip:
    rel = str(clip_dir.relative_to(extract_root))
    meta = ClipMetadata()
    assets: list[PendingAsset] = []

    misc = clip_dir / "misc.txt"
    if misc.is_file():
        meta = parse_misc_txt(misc.read_text(encoding="utf-8"))
    elif (clip_dir / "DE.md").is_file():
        meta.ground_truth_sarcasm = True

    for item in sorted(clip_dir.iterdir()):
        if not item.is_file():
            continue
        if item.name == "misc.txt":
            continue
        if item.suffix.lower() == ".aup3":
            continue

        asset_type = classify_file(item.name)
        if asset_type is None:
            continue

        if asset_type.startswith("transcript") or asset_type == "context_de":
            assets.append(
                PendingAsset(
                    asset_type=asset_type,
                    path=item,
                    content_text=item.read_text(encoding="utf-8"),
                )
            )
        elif asset_type.startswith("audio"):
            assets.append(
                PendingAsset(
                    asset_type=asset_type,
                    path=item,
                    mime_type=mime_for_filename(item.name),
                    content_blob=item.read_bytes(),
                )
            )

    return PendingClip(
        relative_path=rel,
        series_name=series_name,
        metadata=meta,
        assets=assets,
    )


def available_job_pairs(assets: list[PendingAsset]) -> set[tuple[str, str]]:
    types = {a.asset_type for a in assets}
    pairs: set[tuple[str, str]] = set()
    if "transcript_en" in types:
        pairs.add(("text", "en"))
    if "transcript_de" in types:
        pairs.add(("text", "de"))
    if "audio_en" in types:
        pairs.add(("audio", "en"))
    if "audio_de" in types:
        pairs.add(("audio", "de"))
    return pairs


def extract_archive(archive_path: Path, dest: Path) -> None:
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
        return
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest, filter="data")
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def list_archives(raw_data_dir: Path) -> list[Path]:
    archives: list[Path] = []
    if not raw_data_dir.is_dir():
        return archives
    for path in sorted(raw_data_dir.iterdir()):
        lower = path.name.lower()
        if lower.endswith(".zip") or lower.endswith((".tar.gz", ".tgz", ".tar")):
            archives.append(path)
    return archives


def import_archive(
    db: Database,
    conn,
    archive_path: Path,
    model_ids: list[int],
    prompt_ids: list[int],
) -> tuple[int, int]:
    archive_name = archive_path.name
    imported = 0
    jobs_created = 0

    with tempfile.TemporaryDirectory(prefix="sarcasm_import_") as tmp:
        extract_root = Path(tmp)
        extract_archive(archive_path, extract_root)

        series_name = read_series_name(extract_root, archive_name)
        series_id = db.get_or_create_series(conn, series_name)

        for clip_dir in find_clip_directories(extract_root):
            try:
                rel_path = str(clip_dir.relative_to(extract_root))
            except ValueError:
                rel_path = clip_dir.name

            source_key = f"{archive_name}:{rel_path}"
            if db.clip_exists(conn, source_key):
                logger.debug("Skipping existing clip %s", source_key)
                continue

            pending = collect_clip(clip_dir, extract_root, series_name)
            clip_id = db.insert_clip(
                conn,
                series_id=series_id,
                source_key=source_key,
                source_archive=archive_name,
                source_path=rel_path,
                episode=pending.metadata.episode,
                time_start=pending.metadata.time_start,
                time_end=pending.metadata.time_end,
                ground_truth_sarcasm=pending.metadata.ground_truth_sarcasm,
            )

            for asset in pending.assets:
                db.upsert_asset(
                    conn,
                    clip_id=clip_id,
                    asset_type=asset.asset_type,
                    mime_type=asset.mime_type,
                    content_text=asset.content_text,
                    content_blob=asset.content_blob,
                    original_filename=asset.path.name,
                )

            pairs = available_job_pairs(pending.assets)
            jobs_created += db.ensure_jobs_for_clip(
                conn, clip_id, model_ids, prompt_ids, pairs
            )
            imported += 1
            logger.info("Imported clip %s (%d assets)", source_key, len(pending.assets))

    return imported, jobs_created


def sync_models(db: Database, conn, model_names: list[str]) -> list[int]:
    ids: list[int] = []
    for name in model_names:
        ids.append(db.upsert_model(conn, name))
    return ids


def sync_prompts(db: Database, conn, config: Config) -> list[int]:
    prompts = config.load_prompts()
    if not prompts:
        raise ValueError(
            f"No .txt prompt files found in {config.prompts_dir}"
        )
    ids: list[int] = []
    for slug, path in prompts:
        ids.append(db.upsert_prompt(conn, slug, path.name))
    return ids


def ensure_jobs_for_clips(
    db: Database,
    conn,
    model_ids: list[int],
    prompt_ids: list[int],
) -> int:
    created = 0
    rows = conn.execute("SELECT id FROM clips").fetchall()
    for row in rows:
        clip_id = int(row["id"])
        assets = db.get_clip_assets(conn, clip_id)
        pairs: set[tuple[str, str]] = set()
        if "transcript_en" in assets:
            pairs.add(("text", "en"))
        if "transcript_de" in assets:
            pairs.add(("text", "de"))
        if "audio_en" in assets:
            pairs.add(("audio", "en"))
        if "audio_de" in assets:
            pairs.add(("audio", "de"))
        created += db.ensure_jobs_for_clip(
            conn, clip_id, model_ids, prompt_ids, pairs
        )
    return created


def sync_from_config(
    db: Database, conn, config: Config
) -> tuple[list[str], list[int], list[int], int]:
    model_names = config.load_models()
    prompt_ids = sync_prompts(db, conn, config)

    if not model_names:
        return model_names, [], prompt_ids, 0

    model_ids = sync_models(db, conn, model_names)
    created = ensure_jobs_for_clips(db, conn, model_ids, prompt_ids)
    return model_names, model_ids, prompt_ids, created


def sync_models_from_config(
    db: Database, conn, config: Config
) -> tuple[list[str], list[int], int]:
    model_names, model_ids, _, created = sync_from_config(db, conn, config)
    return model_names, model_ids, created


def run_sync_models(config: Config) -> None:
    configure_logging()
    db = Database(config.sqlite_db)
    db.initialize()

    with db.session() as conn:
        model_names, _, prompt_ids, created = sync_from_config(db, conn, config)

    if not model_names:
        logger.error("No models listed in %s", config.models_path)
        return

    logger.info(
        "Synced %d models from %s, %d prompts from %s, created %d new jobs (db: %s)",
        len(model_names),
        config.models_path,
        len(prompt_ids),
        config.prompts_dir,
        created,
        config.sqlite_db,
    )


def run_import(config: Config) -> None:
    configure_logging()
    db = Database(config.sqlite_db)
    db.initialize()

    model_names = config.load_models()
    if not model_names:
        logger.warning("No models listed in %s", config.models_path)

    archives = list_archives(config.raw_data_dir)
    if not archives:
        logger.error("No archives found in %s", config.raw_data_dir)
        return

    total_imported = 0
    total_jobs = 0

    with db.session() as conn:
        _, model_ids, prompt_ids, new_model_jobs = sync_from_config(db, conn, config)
        total_jobs += new_model_jobs

        for archive in archives:
            logger.info("Importing %s", archive.name)
            imported, jobs = import_archive(
                db, conn, archive, model_ids, prompt_ids
            )
            total_imported += imported
            total_jobs += jobs

    logger.info(
        "Import complete: %d new clips, %d new jobs (db: %s)",
        total_imported,
        total_jobs,
        config.sqlite_db,
    )
