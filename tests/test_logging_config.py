from __future__ import annotations

import logging

import pytest

from sarcasm_detector.logging_config import configure_logging


class TestConfigureLogging:
    def test_configure_logging_sets_timestamp_format(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging()
        logging.getLogger("test.logging").info("hello runner")
        err = capsys.readouterr().err
        assert "hello runner" in err
        assert "test.logging:" in err
