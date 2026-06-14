from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Multimodal audio encoders (Qwen2-Audio / Whisper-large-v3) resample input to
# 16 kHz mono and build 128-band mel-spectrograms. Storing 16 kHz mono FLAC is
# lossless relative to that pipeline — unlike MP3/AAC which can alter prosody.
CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1
CANONICAL_MIME = "audio/flac"
CANONICAL_ENCODING = "canonical_flac"

MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}


def mime_for_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return MIME_BY_EXT.get(ext, "application/octet-stream")


def to_canonical_flac(audio_bytes: bytes, source_filename: str) -> bytes:
    """Normalize to 16 kHz mono and encode as FLAC for compact lossless storage."""
    suffix = Path(source_filename).suffix or ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / f"input{suffix}"
        dst = tmp_path / "out.flac"
        src.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(CANONICAL_CHANNELS),
            "-ar",
            str(CANONICAL_SAMPLE_RATE),
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg FLAC encode failed: {stderr}") from exc
        return dst.read_bytes()


def flac_to_wav_pcm(audio_bytes: bytes) -> bytes:
    """Decode canonical FLAC to WAV PCM for Ollama (requires RIFF header)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "input.flac"
        dst = tmp_path / "output.wav"
        src.write_bytes(audio_bytes)
        cmd = ["ffmpeg", "-y", "-i", str(src), "-f", "wav", str(dst)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg FLAC decode failed: {stderr}") from exc
        return dst.read_bytes()


def prepare_audio_for_ollama(
    audio_bytes: bytes,
    *,
    mime_type: str | None,
    blob_encoding: str | None,
    original_filename: str,
) -> tuple[bytes, str]:
    """Return 16 kHz mono WAV bytes for Ollama multimodal input."""
    if blob_encoding == CANONICAL_ENCODING or mime_type == CANONICAL_MIME:
        return flac_to_wav_pcm(audio_bytes), "audio/wav"

    suffix = Path(original_filename).suffix or ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / f"input{suffix}"
        dst = tmp_path / "output.wav"
        src.write_bytes(audio_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(CANONICAL_CHANNELS),
            "-ar",
            str(CANONICAL_SAMPLE_RATE),
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg conversion failed: {stderr}") from exc
        return dst.read_bytes(), "audio/wav"


def encode_audio_base64(
    audio_bytes: bytes,
    *,
    mime_type: str | None,
    blob_encoding: str | None,
    original_filename: str,
) -> str:
    wav_bytes, _ = prepare_audio_for_ollama(
        audio_bytes,
        mime_type=mime_type,
        blob_encoding=blob_encoding,
        original_filename=original_filename,
    )
    return base64.b64encode(wav_bytes).decode("ascii")
