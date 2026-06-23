from __future__ import annotations

import os
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from sarcasm_detector.config import Config
from sarcasm_detector.db import Database
from sarcasm_detector.import_raw import (
    available_job_pairs,
    classify_file,
    collect_clip,
    ensure_jobs_for_clips,
    extract_archive,
    find_clip_directories,
    import_archive,
    is_clip_directory,
    list_archives,
    parse_misc_txt,
    read_series_name,
    run_import,
    run_sync_models,
    sync_models,
)


class TestParseMiscTxt:
    def test_parses_all_fields(self) -> None:
        meta = parse_misc_txt(
            "IS_SARCASM=True\nEPISODE=S01E01\nTIME_START=01:00\nTIME_END=02:00\n"
        )
        assert meta.ground_truth_sarcasm is True
        assert meta.episode == "S01E01"
        assert meta.time_start == "01:00"
        assert meta.time_end == "02:00"

    def test_parses_film_and_false(self) -> None:
        meta = parse_misc_txt("IS_SARCASM=false\nFILM=Deadpool 2\n")
        assert meta.ground_truth_sarcasm is False
        assert meta.episode == "Deadpool 2"

    def test_skips_invalid_lines(self) -> None:
        meta = parse_misc_txt("not a kv\nIS_SARCASM=yes\n")
        assert meta.ground_truth_sarcasm is True


class TestClassifyFile:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("en.txt", "transcript_en"),
            ("en.txt.txt", "transcript_en"),
            ("de.txt", "transcript_de"),
            ("DE.md", "transcript_de"),
            ("SPEC.md", "context_de"),
            ("en.audio.mp3", "audio_en"),
            ("en.audio12.mp3", "audio_en"),
            ("de.audio.wav", "audio_de"),
            ("DE.m4a", "audio_de"),
            ("readme.txt", None),
        ],
    )
    def test_classify(self, filename: str, expected: str | None) -> None:
        assert classify_file(filename) == expected


class TestClipDiscovery:
    def test_is_clip_directory(self, tmp_path: Path) -> None:
        assert not is_clip_directory(tmp_path)
        misc = tmp_path / "clip"
        misc.mkdir()
        (misc / "misc.txt").write_text("IS_SARCASM=true\n")
        assert is_clip_directory(misc)

        werner = tmp_path / "werner"
        werner.mkdir()
        (werner / "DE.md").write_text("text")
        assert is_clip_directory(werner)

    def test_find_clip_directories_skips_aup3(self, tmp_path: Path) -> None:
        clip = tmp_path / "01"
        clip.mkdir()
        (clip / "misc.txt").write_text("IS_SARCASM=true\n")
        aup3 = tmp_path / "aup3" / "01"
        aup3.mkdir(parents=True)
        (aup3 / "misc.txt").write_text("IS_SARCASM=true\n")
        found = find_clip_directories(tmp_path)
        assert clip in found
        assert aup3 not in found

    def test_collect_clip_bilingual(self, tmp_path: Path) -> None:
        clip = tmp_path / "Show" / "01"
        clip.mkdir(parents=True)
        (clip / "misc.txt").write_text("IS_SARCASM=true\nEPISODE=S01E01\n")
        (clip / "en.txt").write_text("English")
        (clip / "de.txt").write_text("German")
        (clip / "en.audio.mp3").write_bytes(b"mp3")
        (clip / "de.audio.mp3").write_bytes(b"mp3")

        pending = collect_clip(clip, tmp_path, "Show")
        assert pending.metadata.ground_truth_sarcasm is True
        assert len(pending.assets) == 4
        pairs = available_job_pairs(pending.assets)
        assert pairs == {("text", "en"), ("text", "de"), ("audio", "en"), ("audio", "de")}

    def test_collect_werner_defaults_sarcasm(self, tmp_path: Path) -> None:
        clip = tmp_path / "Werner" / "01"
        clip.mkdir(parents=True)
        (clip / "DE.md").write_text("**Werner:** hi")
        (clip / "DE.m4a").write_bytes(b"m4a")
        pending = collect_clip(clip, tmp_path, "Werner")
        assert pending.metadata.ground_truth_sarcasm is True


class TestArchives:
    def test_extract_zip(self, sample_zip: Path, tmp_path: Path) -> None:
        dest = tmp_path / "out"
        dest.mkdir()
        extract_archive(sample_zip, dest)
        assert (dest / "Family Guy" / "01" / "en.txt").is_file()

    def test_extract_unsupported(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.bin"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_archive(bad, tmp_path / "out")

    def test_list_archives(self, raw_data_dir: Path) -> None:
        archives = list_archives(raw_data_dir)
        assert len(archives) == 1
        assert not list_archives(raw_data_dir / "missing")

    def test_read_series_name_override(self, tmp_path: Path) -> None:
        assert read_series_name(tmp_path, "Deadpool_Clips.zip") == "Deadpool"

    def test_read_series_name_from_file(self, tmp_path: Path) -> None:
        (tmp_path / "series.txt").write_text("Custom Show\n")
        assert read_series_name(tmp_path, "unknown.zip") == "Custom Show"

    def test_read_series_name_from_stem(self, tmp_path: Path) -> None:
        assert read_series_name(tmp_path, "my-show.zip") == "my show"


class TestImportArchive:
    def test_import_and_idempotent(self, tmp_db, sample_zip: Path) -> None:
        with tmp_db.session() as conn:
            model_id = tmp_db.upsert_model(conn, "m1")
            prompt_id = tmp_db.upsert_prompt(conn, "default", "default.txt")
            imported, jobs = import_archive(
                tmp_db, conn, sample_zip, [model_id], [prompt_id]
            )
            assert imported == 1
            assert jobs == 4
            imported2, jobs2 = import_archive(
                tmp_db, conn, sample_zip, [model_id], [prompt_id]
            )
            assert imported2 == 0
            assert jobs2 == 0

    def test_import_deadpool(self, tmp_db, deadpool_zip: Path) -> None:
        with tmp_db.session() as conn:
            model_id = tmp_db.upsert_model(conn, "m1")
            prompt_id = tmp_db.upsert_prompt(conn, "default", "default.txt")
            imported, _ = import_archive(
                tmp_db, conn, deadpool_zip, [model_id], [prompt_id]
            )
            assets = tmp_db.get_clip_assets(
                conn,
                conn.execute("SELECT id FROM clips").fetchone()["id"],
            )
        assert imported == 1
        assert "transcript_en" in assets
        assert assets["transcript_en"]["content_text"].startswith("Speaker")

    def test_import_werner_tar(self, tmp_db, werner_tar: Path) -> None:
        with tmp_db.session() as conn:
            model_id = tmp_db.upsert_model(conn, "m1")
            prompt_id = tmp_db.upsert_prompt(conn, "default", "default.txt")
            imported, jobs = import_archive(
                tmp_db, conn, werner_tar, [model_id], [prompt_id]
            )
        assert imported == 1
        assert jobs == 2

    def test_import_creates_jobs_per_prompt(self, tmp_db, sample_zip: Path) -> None:
        with tmp_db.session() as conn:
            model_id = tmp_db.upsert_model(conn, "m1")
            prompt_a = tmp_db.upsert_prompt(conn, "a", "a.txt")
            prompt_b = tmp_db.upsert_prompt(conn, "b", "b.txt")
            _, jobs = import_archive(
                tmp_db, conn, sample_zip, [model_id], [prompt_a, prompt_b]
            )
        assert jobs == 8

    def test_run_import_no_archives(
        self, config: Config, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config.raw_data_dir.mkdir()
        run_import(config)
        assert "No archives found" in capsys.readouterr().err

    def test_run_import_no_models_warning(
        self, config: Config, sample_zip: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config.raw_data_dir.mkdir()
        config.models_path.write_text("# only comments\n")
        import shutil

        shutil.copy(sample_zip, config.raw_data_dir / sample_zip.name)
        run_import(config)
        assert "No models listed" in capsys.readouterr().err

    def test_collect_skips_unrecognized_and_aup3(self, tmp_path: Path) -> None:
        clip = tmp_path / "01"
        clip.mkdir()
        (clip / "misc.txt").write_text("IS_SARCASM=true\n")
        (clip / "en.txt").write_text("text")
        (clip / "notes.txt").write_text("ignored")
        (clip / "project.aup3").write_bytes(b"aup3")
        pending = collect_clip(clip, tmp_path, "Show")
        types = {a.asset_type for a in pending.assets}
        assert types == {"transcript_en"}

    def test_run_import_full(self, config: Config, sample_zip: Path) -> None:
        import shutil

        config.raw_data_dir.mkdir()
        shutil.copy(sample_zip, config.raw_data_dir / sample_zip.name)
        run_import(config)
        db = Database(config.sqlite_db)
        with db.session() as conn:
            assert db.count_clips(conn) == 1
            assert db.job_status_counts(conn)["pending"] == 4

    def test_ensure_jobs_for_clips(self, tmp_db, sample_zip: Path) -> None:
        with tmp_db.session() as conn:
            model_a = tmp_db.upsert_model(conn, "model-a")
            prompt_id = tmp_db.upsert_prompt(conn, "default", "default.txt")
            import_archive(tmp_db, conn, sample_zip, [model_a], [prompt_id])
            model_b = tmp_db.upsert_model(conn, "model-b")
            created = ensure_jobs_for_clips(tmp_db, conn, [model_b], [prompt_id])
            assert created == 4
            assert sync_models(tmp_db, conn, ["model-a", "model-b"]) == [model_a, model_b]

    def test_run_sync_models(self, config: Config, sample_zip: Path) -> None:
        import shutil

        config.raw_data_dir.mkdir()
        shutil.copy(sample_zip, config.raw_data_dir / sample_zip.name)
        run_import(config)
        (config.models_path).write_text("test-model\nnew-model\n")
        run_sync_models(config)
        db = Database(config.sqlite_db)
        with db.session() as conn:
            assert db.job_status_counts(conn)["pending"] == 8
