"""Experiment worker process executed inside one GNU Screen session."""

from __future__ import annotations

import argparse
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from retrieval_core.cli import main as stage_main
from experiment_models import (
    TERMINAL_STATES,
    load_plan,
    read_status,
    status_path,
    update_status,
)
from screen import session_exists
from retrieval_core.utils.hashing import sha256_text
from retrieval_core.utils.time import utc_now


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Execute one prepared experiment run.")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--lost-grace-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    raise SystemExit(
        run_worker(
            args.experiment_dir,
            args.run_name,
            poll_seconds=args.poll_seconds,
            lost_grace_seconds=args.lost_grace_seconds,
        )
    )


def run_worker(
    experiment_dir: Path,
    run_name: str,
    *,
    poll_seconds: float,
    lost_grace_seconds: float,
) -> int:
    directory = experiment_dir.expanduser().resolve()
    plan = load_plan(directory)
    run = next((candidate for candidate in plan.runs if candidate.name == run_name), None)
    if run is None:
        raise KeyError(f"Experiment {plan.experiment_id!r} has no run named {run_name!r}.")
    own_status_path = status_path(directory, run)
    initial_status = read_status(own_status_path)
    wait_for = initial_status.get("wait_for")

    try:
        if isinstance(wait_for, dict):
            update_status(own_status_path, state="waiting", waiting_since=utc_now())
            predecessor_state = wait_for_predecessor(
                wait_for,
                poll_seconds=poll_seconds,
                lost_grace_seconds=lost_grace_seconds,
            )
            update_status(
                own_status_path,
                predecessor_terminal_state=predecessor_state,
                wait_finished_at=utc_now(),
            )

        config_path = directory / run.config_file
        config_text = config_path.read_text(encoding="utf-8")
        if sha256_text(config_text) != run.config_sha256:
            raise ValueError(f"Prepared config checksum changed for {run.name!r}: {config_path}")

        update_status(
            own_status_path,
            state="running",
            started_at=utc_now(),
            finished_at=None,
            exit_code=None,
        )
        project_root = Path(plan.project_root)
        config_dir = config_path.parent
        config_name = Path(run.config_file).stem
        previous_cwd = Path.cwd()
        try:
            # Project-local config discovery and relative artifact paths depend on this cwd.
            import os

            os.chdir(project_root)
            stage_main(["--config-dir", str(config_dir), config_name])
        finally:
            os.chdir(previous_cwd)
    except BaseException as exc:
        traceback.print_exc()
        exit_code = (
            exc.code
            if isinstance(exc, SystemExit) and isinstance(exc.code, int)
            else 0
            if isinstance(exc, SystemExit) and exc.code is None
            else 1
        )
        update_status(
            own_status_path,
            state="failed",
            finished_at=utc_now(),
            exit_code=exit_code,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return exit_code

    update_status(
        own_status_path,
        state="succeeded",
        finished_at=utc_now(),
        exit_code=0,
    )
    return 0


def wait_for_predecessor(
    predecessor: dict[str, Any],
    *,
    poll_seconds: float,
    lost_grace_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    session_exists_fn: Callable[[str], bool] = session_exists,
) -> str:
    predecessor_status_path = Path(str(predecessor["status_path"]))
    predecessor_screen = str(predecessor["screen_name"])
    missing_since: float | None = None

    while True:
        state = str(read_status(predecessor_status_path).get("state", ""))
        if state in TERMINAL_STATES:
            return state

        if session_exists_fn(predecessor_screen):
            missing_since = None
        else:
            now = monotonic_fn()
            missing_since = now if missing_since is None else missing_since
            if now - missing_since >= lost_grace_seconds:
                return "lost"

        sleep_fn(poll_seconds)
if __name__ == "__main__":
    main()
