"""Small, testable GNU Screen command adapter."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

SCREEN_SESSION_PATTERN = re.compile(r"^\s*\d+\.([^\s]+)\s+\(", re.MULTILINE)


def require_screen() -> str:
    executable = shutil.which("screen")
    if executable is None:
        raise SystemExit(
            "GNU Screen is required to launch experiments. Install the 'screen' package first."
        )
    return executable


def list_screen_sessions(*, executable: str = "screen") -> set[str]:
    result = subprocess.run(
        [executable, "-ls"],
        capture_output=True,
        check=False,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    return set(SCREEN_SESSION_PATTERN.findall(output))


def launch_screen(
    *,
    session_name: str,
    log_file: Path,
    command: Sequence[str],
    cwd: Path,
    executable: str = "screen",
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "-L",
            "-Logfile",
            str(log_file),
            "-dmS",
            session_name,
            *command,
        ],
        cwd=cwd,
        check=True,
    )


def session_exists(session_name: str, *, executable: str = "screen") -> bool:
    return session_name in list_screen_sessions(executable=executable)
