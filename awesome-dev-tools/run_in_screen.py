"""Launch one arbitrary command in a detached GNU Screen session."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from _internal.screen import launch_screen, list_screen_sessions, require_screen

SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch one command in a new detached GNU Screen session."
    )
    parser.add_argument("--name", help="Screen session name; generated when omitted.")
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Command working directory; defaults to the current directory.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Screen log path; defaults to artifacts/screens/<session>.log below --cwd.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if not sys.platform.startswith("linux"):
        raise SystemExit("run-in-screen requires Linux because it launches GNU Screen.")

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("Provide a command after '--'.")

    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise SystemExit(f"Command working directory does not exist: {cwd}")

    session_name = args.name or default_session_name(command)
    validate_session_name(session_name)
    screen_executable = require_screen()
    if session_name in list_screen_sessions(executable=screen_executable):
        raise SystemExit(f"Screen session already exists: {session_name}")

    log_file = resolve_log_file(args.log_file, cwd=cwd, session_name=session_name)
    launch_screen(
        session_name=session_name,
        log_file=log_file,
        command=command,
        cwd=cwd,
        executable=screen_executable,
    )

    print(f"Launched detached screen: {session_name}")
    print(f"Log: {log_file}")
    print(f"Attach with: screen -r {session_name}")


def default_session_name(
    command: Sequence[str],
    *,
    timestamp: str | None = None,
) -> str:
    """Generate a readable, collision-resistant Screen session name."""

    stem = "command"
    if len(command) >= 4 and list(command[:3]) == ["uv", "run", "stage"]:
        stem = f"stage-{command[3]}"
    elif command:
        stem = Path(command[0]).stem or stem
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-._") or "command"
    suffix = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{normalized}-{suffix}"[:75].rstrip("-._")


def validate_session_name(session_name: str) -> None:
    """Reject names that GNU Screen cannot address reliably."""

    if not session_name or not SESSION_NAME_PATTERN.fullmatch(session_name):
        raise SystemExit(
            "Screen session names may contain only letters, numbers, '.', '_', and '-'."
        )


def resolve_log_file(
    log_file: Path | None,
    *,
    cwd: Path,
    session_name: str,
) -> Path:
    """Resolve the explicit or default Screen log path."""

    if log_file is None:
        return cwd / "artifacts" / "screens" / f"{session_name}.log"
    expanded = log_file.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (cwd / expanded).resolve()


if __name__ == "__main__":
    main()
