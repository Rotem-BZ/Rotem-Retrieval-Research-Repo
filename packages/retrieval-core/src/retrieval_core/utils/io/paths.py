"""Path resolution and directory creation helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

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


def find_git_root(*, working_dir: str | Path | None = None) -> Path:
    """Return the absolute root of the Git repository containing the working directory."""

    base = Path(working_dir).resolve() if working_dir is not None else _original_cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=base,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to resolve paths.repo_root.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or f"exit code {exc.returncode}"
        raise RuntimeError(f"Could not resolve paths.repo_root from {base}: {detail}") from exc

    root = Path(result.stdout.strip()).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Git returned an invalid repository root: {root}")
    return root


def ensure_parent(path: str | Path) -> Path:
    """Resolve a path and create its parent directory."""

    resolved = project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _original_cwd() -> Path:
    try:
        return Path(get_original_cwd()).resolve()
    except ValueError:
        return Path.cwd().resolve()
