"""Path resolution and directory creation helpers."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import get_original_cwd


def project_path(path: str | Path) -> Path:
    """Resolve a path relative to the original working directory Hydra saw."""

    candidate = Path(str(path))
    if candidate.is_absolute():
        return candidate

    try:
        base = Path(get_original_cwd())
    except ValueError:
        base = Path.cwd()
    return base / candidate


def ensure_parent(path: str | Path) -> Path:
    """Resolve a path and create its parent directory."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Resolve and create a directory."""

    resolved = project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
