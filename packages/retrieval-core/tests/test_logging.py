from __future__ import annotations

import logging
from pathlib import Path

from retrieval_core.utils.logging import (
    CONSOLE_LOG_LEVEL,
    RUN_FILE_LOG_LEVEL,
    configure_console_logging,
    run_file_logging,
)


def test_console_logging_is_idempotent() -> None:
    first = configure_console_logging()
    second = configure_console_logging()

    assert first is second
    assert first.level == CONSOLE_LOG_LEVEL


def test_run_file_logging_captures_debug_and_removes_handler(tmp_path: Path) -> None:
    logger = logging.getLogger("retrieval_core.tests.logging")
    path = tmp_path / "run.log"

    with run_file_logging(path):
        logger.debug("debug details")
        logger.info("visible event")

    contents = path.read_text(encoding="utf-8")
    assert "DEBUG retrieval_core.tests.logging debug details" in contents
    assert "INFO retrieval_core.tests.logging visible event" in contents
    assert not any(
        handler.get_name() == f"retrieval-run-file:{path}"
        for handler in logging.getLogger().handlers
    )
    assert RUN_FILE_LOG_LEVEL == logging.DEBUG
