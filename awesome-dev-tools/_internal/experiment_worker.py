"""Execute one explicit experiment run inside a GNU Screen session."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _internal.experiment_models import (
    TERMINAL_STATES,
    load_plan,
    read_status,
    render_hydra_command,
    status_path,
    update_status,
)
from retrieval_core.cli import main as stage_main
from retrieval_core.utils.config import resolve_config_entrypoint
from retrieval_core.utils.time import utc_now
from _internal.screen import session_exists


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Execute one experiment run definition."
    )
    parser.add_argument("--entrypoint", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--lost-grace-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    raise SystemExit(
        run_worker(
            args.entrypoint,
            poll_seconds=args.poll_seconds,
            lost_grace_seconds=args.lost_grace_seconds,
        )
    )


def run_worker(
    entrypoint: Path,
    *,
    poll_seconds: float,
    lost_grace_seconds: float,
) -> int:
    resolved = resolve_config_entrypoint(entrypoint)
    if resolved.experiment_dir is None or not resolved.config_name.startswith("runs/"):
        raise ValueError(
            f"Experiment worker requires a configs/runs/*.yaml entrypoint: {entrypoint}"
        )
    directory = resolved.experiment_dir
    run_name = Path(resolved.config_name).name
    plan = load_plan(directory)
    run = next(
        (candidate for candidate in plan.runs if candidate.name == run_name), None
    )
    if run is None:
        raise KeyError(
            f"Experiment {plan.experiment_id!r} has no run named {run_name!r}."
        )
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

        update_status(
            own_status_path,
            state="running",
            started_at=utc_now(),
            finished_at=None,
            exit_code=None,
        )
        print("Hydra command:")
        print(render_hydra_command(run), flush=True)
        previous_cwd = Path.cwd()
        try:
            os.chdir(plan.project_root)
            stage_main(
                [
                    run.stage_name,
                    "--entrypoint",
                    str(run.definition_file),
                ]
            )
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
