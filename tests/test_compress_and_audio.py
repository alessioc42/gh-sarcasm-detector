from __future__ import annotations

import shutil
from unittest import mock

import pytest

from sarcasm_detector.audio_util import (
    CANONICAL_ENCODING,
    CANONICAL_MIME,
    encode_audio_base64,
    flac_to_wav_pcm,
    mime_for_filename,
    prepare_audio_for_ollama,
    to_canonical_flac,
)
from sarcasm_detector.compress import _flac_filename, run_compress
from sarcasm_detector.config import Config


class TestMimeForFilename:
    @pytest.mark.parametrize(
        "name,mime",
        [
            ("a.mp3", "audio/mpeg"),
            ("a.wav", "audio/wav"),
            ("a.flac", "audio/flac"),
            ("a.unknown", "application/octet-stream"),
        ],
    )
    def test_mime(self, name: str, mime: str) -> None:
        assert mime_for_filename(name) == mime


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestAudioUtilFfmpeg:
    @staticmethod
    def _make_wav_bytes() -> bytes:
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tone.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.2",
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            return out.read_bytes()

    def test_to_canonical_flac_roundtrip(self) -> None:
        wav = self._make_wav_bytes()
        flac = to_canonical_flac(wav, "tone.wav")
        assert flac[:4] == b"fLaC"
        pcm = flac_to_wav_pcm(flac)
        assert pcm[:4] == b"RIFF"

    def test_prepare_raw_vs_canonical(self) -> None:
        wav = self._make_wav_bytes()
        flac = to_canonical_flac(wav, "tone.wav")
        raw_out, _ = prepare_audio_for_ollama(
            wav, mime_type="audio/wav", blob_encoding="raw", original_filename="tone.wav"
        )
        canon_out, _ = prepare_audio_for_ollama(
            flac,
            mime_type=CANONICAL_MIME,
            blob_encoding=CANONICAL_ENCODING,
            original_filename="tone.flac",
        )
        assert raw_out[:4] == b"RIFF"
        assert canon_out[:4] == b"RIFF"

    def test_encode_audio_base64(self) -> None:
        wav = self._make_wav_bytes()
        b64 = encode_audio_base64(
            wav, mime_type="audio/wav", blob_encoding="raw", original_filename="tone.wav"
        )
        assert isinstance(b64, str)
        assert len(b64) > 0


class TestAudioUtilErrors:
    def test_to_canonical_flac_invalid(self) -> None:
        with pytest.raises(RuntimeError, match="ffmpeg"):
            to_canonical_flac(b"not-audio", "bad.bin")

    @mock.patch("sarcasm_detector.audio_util.subprocess.run")
    def test_prepare_raw_conversion_error(self, mock_run: mock.Mock) -> None:
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"bad wav"
        )
        with pytest.raises(RuntimeError, match="ffmpeg conversion failed"):
            prepare_audio_for_ollama(
                b"bad",
                mime_type="audio/wav",
                blob_encoding="raw",
                original_filename="x.wav",
            )

    @mock.patch("sarcasm_detector.audio_util.subprocess.run")
    def test_flac_decode_error(self, mock_run: mock.Mock) -> None:
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"failed"
        )
        with pytest.raises(RuntimeError, match="FLAC decode"):
            flac_to_wav_pcm(b"fLaC" + b"\x00" * 10)


class TestCompress:
    def test_flac_filename(self) -> None:
        assert _flac_filename("en.audio.mp3") == "en.audio.flac"

    @mock.patch("sarcasm_detector.compress.to_canonical_flac", return_value=b"fLaC-data")
    def test_run_compress(
        self, mock_flac: mock.Mock, config_with_db: Config, tmp_db
    ) -> None:
        config = config_with_db
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            tmp_db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type="audio_en",
                mime_type="audio/mpeg",
                content_text=None,
                content_blob=b"raw-audio",
                blob_encoding="raw",
                original_filename="en.mp3",
            )

        run_compress(config)
        mock_flac.assert_called_once_with(b"raw-audio", "en.mp3")

        with tmp_db.session() as conn:
            row = conn.execute(
                "SELECT blob_encoding, mime_type, original_filename FROM clip_assets"
            ).fetchone()
        assert row["blob_encoding"] == CANONICAL_ENCODING
        assert row["mime_type"] == CANONICAL_MIME
        assert row["original_filename"] == "en.flac"

    @mock.patch("sarcasm_detector.compress.to_canonical_flac")
    def test_run_compress_skips_canonical(
        self, mock_flac: mock.Mock, config_with_db: Config, tmp_db
    ) -> None:
        config = config_with_db
        with tmp_db.session() as conn:
            series_id = tmp_db.get_or_create_series(conn, "Show")
            clip_id = tmp_db.insert_clip(
                conn,
                series_id=series_id,
                source_key="k",
                source_archive="a.zip",
                source_path="01",
                episode=None,
                time_start=None,
                time_end=None,
                ground_truth_sarcasm=True,
            )
            tmp_db.upsert_asset(
                conn,
                clip_id=clip_id,
                asset_type="audio_en",
                mime_type=CANONICAL_MIME,
                content_text=None,
                content_blob=b"fLaC",
                blob_encoding=CANONICAL_ENCODING,
                original_filename="en.flac",
            )

        run_compress(config)
        mock_flac.assert_not_called()
