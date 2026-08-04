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


def test_console_logging_formats_event_fields_on_separate_lines() -> None:
    handler = configure_console_logging()
    record = logging.LogRecord(
        name="retrieval_core.tests.logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Stage started: stage=indexing run_id=test-run output_dir=C:/some path",
        args=(),
        exc_info=None,
    )

    rendered = handler.format(record)

    assert "INFO     retrieval_core.tests.logging\n" in rendered
    assert "  Stage started:\n" in rendered
    assert "    stage=indexing\n" in rendered
    assert "    run_id=test-run\n" in rendered
    assert "    output_dir=C:/some path" in rendered


def test_console_logging_colors_level_labels_only_for_terminals() -> None:
    handler = configure_console_logging()
    record = logging.LogRecord(
        name="retrieval_core.tests.logging",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Something needs attention",
        args=(),
        exc_info=None,
    )
    record.terminal_colors = True

    rendered = handler.format(record)

    assert "\033[33mWARNING \033[0m" in rendered
    assert "\033[33mSomething needs attention" not in rendered


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
