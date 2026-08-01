"""Repository-wide standard-library logging policy."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

CONSOLE_LOG_LEVEL = logging.INFO
RUN_FILE_LOG_LEVEL = logging.DEBUG
THIRD_PARTY_LOG_LEVEL = logging.WARNING
RUN_LOG_FILENAME = "run.log"

_CONSOLE_HANDLER_NAME = "retrieval-console"
_FIRST_PARTY_LOGGERS = (
    "retrieval_core",
    "retrieval_components",
    "sparse_autoencoder_retrieval",
    "query_repetition",
    "_internal",
)
_THIRD_PARTY_LOGGERS = (
    "haystack",
    "httpx",
    "requests",
    "sentence_transformers",
    "torch",
    "transformers",
    "urllib3",
)


class _UtcFormatter(logging.Formatter):
    """Format timestamps in UTC without changing process-wide timezone state."""

    converter = time.gmtime


class _StderrHandler(logging.StreamHandler):
    """Resolve stderr at emit time so capture contexts cannot leave a stale stream."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def _formatter() -> logging.Formatter:
    return _UtcFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _configure_logger_levels() -> None:
    for name in _FIRST_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(RUN_FILE_LOG_LEVEL)
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(THIRD_PARTY_LOG_LEVEL)


def configure_console_logging() -> logging.Handler:
    """Install the repository-owned stderr handler exactly once."""

    _configure_logger_levels()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if handler.get_name() == _CONSOLE_HANDLER_NAME:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.stderr
            return handler

    handler = _StderrHandler()
    handler.set_name(_CONSOLE_HANDLER_NAME)
    handler.setLevel(CONSOLE_LOG_LEVEL)
    handler.setFormatter(_formatter())
    root_logger.addHandler(handler)
    return handler


@contextmanager
def run_file_logging(path: str | Path) -> Iterator[Path]:
    """Attach a DEBUG file handler for the lifetime of one immutable stage run."""

    _configure_logger_levels()
    resolved = Path(path)
    if not resolved.parent.is_dir():
        raise FileNotFoundError(f"Run log directory does not exist: {resolved.parent}")

    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.set_name(f"retrieval-run-file:{resolved}")
    handler.setLevel(RUN_FILE_LOG_LEVEL)
    handler.setFormatter(_formatter())
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield resolved
    finally:
        root_logger.removeHandler(handler)
        handler.close()
