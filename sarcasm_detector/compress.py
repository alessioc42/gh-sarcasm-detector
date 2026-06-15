from __future__ import annotations

import logging
from pathlib import Path

from .audio_util import (
    CANONICAL_ENCODING,
    CANONICAL_MIME,
    to_canonical_flac,
)
from .config import Config
from .db import Database
from .logging_config import configure_logging

logger = logging.getLogger(__name__)


def _flac_filename(filename: str) -> str:
    return f"{Path(filename).stem}.flac"


def run_compress(config: Config) -> None:
    """Re-encode stored audio as 16 kHz mono FLAC (canonical for LLM encoders)."""
    configure_logging()
    db = Database(config.sqlite_db)
    db.initialize()

    compressed = 0
    skipped = 0
    bytes_before = 0
    bytes_after = 0

    with db.session() as conn:
        rows = conn.execute(
            """
            SELECT id, content_blob, original_filename, blob_encoding
            FROM clip_assets
            WHERE asset_type LIKE 'audio_%'
              AND content_blob IS NOT NULL
            """
        ).fetchall()

        for row in rows:
            if row["blob_encoding"] == CANONICAL_ENCODING:
                skipped += 1
                continue

            asset_id = int(row["id"])
            raw = bytes(row["content_blob"])
            filename = str(row["original_filename"])
            bytes_before += len(raw)

            flac = to_canonical_flac(raw, filename)
            bytes_after += len(flac)

            conn.execute(
                """
                UPDATE clip_assets
                SET content_blob = ?,
                    mime_type = ?,
                    blob_encoding = ?,
                    original_filename = ?
                WHERE id = ?
                """,
                (
                    flac,
                    CANONICAL_MIME,
                    CANONICAL_ENCODING,
                    _flac_filename(filename),
                    asset_id,
                ),
            )
            compressed += 1
            logger.info(
                "Compressed asset %d (%s): %d KB -> %d KB",
                asset_id,
                filename,
                len(raw) // 1024,
                len(flac) // 1024,
            )

    with db.connect() as conn:
        conn.execute("VACUUM")
        conn.commit()

    saved = max(0, bytes_before - bytes_after)
    logger.info(
        "Compress complete: %d assets compressed, %d already canonical, "
        "saved ~%.1f MB (%.1f MB -> %.1f MB)",
        compressed,
        skipped,
        saved / 1024 / 1024,
        bytes_before / 1024 / 1024,
        bytes_after / 1024 / 1024,
    )
