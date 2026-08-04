"""Repository-wide standard-library logging policy."""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

CONSOLE_LOG_LEVEL = logging.INFO
RUN_FILE_LOG_LEVEL = logging.DEBUG
THIRD_PARTY_LOG_LEVEL = logging.WARNING
RUN_LOG_FILENAME = "run.log"

_CONSOLE_HANDLER_NAME = "retrieval-console"
_ANSI_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
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


class _ConsoleFormatter(_UtcFormatter):
    """Render terminal events as a compact header followed by readable fields."""

    _FIELD_BOUNDARY = re.compile(r"\s+(?=[A-Za-z_][\w.-]*=)")

    def format(self, record: logging.LogRecord) -> str:
        header = self.formatTime(record, self.datefmt)
        message = record.getMessage()
        summary, separator, fields = message.partition(": ")
        level = f"{record.levelname:<8}"
        if getattr(record, "terminal_colors", False):
            color = _LEVEL_COLORS.get(record.levelno)
            if color:
                level = f"{color}{level}{_ANSI_RESET}"
        lines = [f"{header}  {level} {record.name}"]
        lines.append(f"  {summary}{':' if separator else ''}")
        if separator:
            lines.extend(
                f"    {field}" for field in self._FIELD_BOUNDARY.split(fields)
            )
        if record.exc_info:
            lines.append(self.formatException(record.exc_info))
        if record.stack_info:
            lines.append(self.formatStack(record.stack_info))
        return "\n".join(lines)


class _StderrHandler(logging.StreamHandler):
    """Write around active tqdm bars and resolve stderr at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            record.terminal_colors = _supports_color(sys.stderr)
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def _formatter() -> logging.Formatter:
    return _UtcFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


def _console_formatter() -> logging.Formatter:
    return _ConsoleFormatter(datefmt="%Y-%m-%dT%H:%M:%SZ")


def _supports_color(stream: object) -> bool:
    """Return whether ANSI colors should be emitted to this terminal stream."""

    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


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
    handler.setFormatter(_console_formatter())
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
