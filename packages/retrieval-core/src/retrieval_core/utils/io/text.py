"""Plain-text file writers."""

from __future__ import annotations

from pathlib import Path

from retrieval_core.utils.io.paths import ensure_parent


def write_text(path: str | Path, text: str) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(text, encoding="utf-8")
    return resolved
