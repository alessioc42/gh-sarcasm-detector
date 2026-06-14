from __future__ import annotations

from unittest import mock

import pytest

from sarcasm_detector.__main__ import main


class TestMain:
    def test_help(self) -> None:
        assert main(["--help"]) == 0

    def test_no_args(self) -> None:
        assert main([]) == 1

    def test_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["bogus"]) == 1
        assert "Unknown command" in capsys.readouterr().err

    @mock.patch("sarcasm_detector.__main__.run_import")
    def test_import_command(self, mock_run: mock.Mock) -> None:
        assert main(["import"]) == 0
        mock_run.assert_called_once()

    @mock.patch("sarcasm_detector.__main__.run_compress")
    def test_compress_command(self, mock_run: mock.Mock) -> None:
        assert main(["compress"]) == 0
        mock_run.assert_called_once()

    @mock.patch("sarcasm_detector.__main__.run_jobs")
    def test_run_command(self, mock_run: mock.Mock) -> None:
        assert main(["run"]) == 0
        mock_run.assert_called_once()

    @mock.patch("sarcasm_detector.__main__.run_status")
    def test_status_command(self, mock_run: mock.Mock) -> None:
        assert main(["status"]) == 0
        mock_run.assert_called_once()
