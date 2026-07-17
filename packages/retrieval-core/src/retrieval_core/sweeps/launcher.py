"""Interactive, launch-only GNU Screen sweep queue builder."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from retrieval_core.sweeps.models import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    SweepPlan,
    SweepRun,
    load_plan,
    log_path,
    read_status,
    screen_name,
    status_path,
)
from retrieval_core.sweeps.screen import launch_screen, list_screen_sessions, require_screen
from retrieval_core.utils.io import read_json, write_json_atomic
from retrieval_core.utils.time import utc_now

REGISTRY_SCHEMA_VERSION = 1
SELECTION_RANGE = re.compile(r"^(\d+)-(\d+)$")

STATE_LABELS = {
    "ready": ("○", "READY", ""),
    "launching": ("◌", "LAUNCHING", "yellow"),
    "waiting": ("◷", "WAITING", "yellow"),
    "running": ("●", "RUNNING", "cyan"),
    "succeeded": ("✓", "SUCCEEDED", "green"),
    "failed": ("✗", "FAILED", "red"),
    "launch_failed": ("✗", "LAUNCH FAILED", "red"),
    "lost": ("?", "LOST", "red"),
    "partial": ("!", "PARTIAL", "red"),
    "cancelled": ("-", "CANCELLED", "red"),
}
ANSI = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Select prepared runs, launch their GNU Screen sessions, and exit."
    )
    parser.add_argument("sweep_dir", nargs="?", type=Path)
    parser.add_argument("--select", dest="selection")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--lost-grace-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    if not sys.platform.startswith("linux"):
        raise SystemExit("run-sweep requires Linux because it launches GNU Screen sessions.")
    screen_executable = require_screen()
    launch_interactively(
        sweep_dir=args.sweep_dir,
        selection=args.selection,
        max_parallel=args.max_parallel,
        poll_seconds=args.poll_seconds,
        lost_grace_seconds=args.lost_grace_seconds,
        screen_executable=screen_executable,
    )


def launch_interactively(
    *,
    sweep_dir: Path | None,
    selection: str | None,
    max_parallel: int | None,
    poll_seconds: float,
    lost_grace_seconds: float,
    screen_executable: str,
) -> list[str]:
    directory = choose_sweep_dir(sweep_dir)
    plan = load_plan(directory)
    sessions = list_screen_sessions(executable=screen_executable)
    states = {run.index: run_state(directory, run, sessions) for run in plan.runs}
    print_run_table(plan, states)

    selection_text = selection
    if selection_text is None:
        while True:
            selection_text = input("Select runs [ready]: ").strip() or "ready"
            try:
                selected_indices = parse_selection(selection_text, plan.runs, states)
            except ValueError as exc:
                print(exc)
                continue
            break
    else:
        try:
            selected_indices = parse_selection(selection_text, plan.runs, states)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    selected_runs = [run for run in plan.runs if run.index in selected_indices]
    eligible_runs = [
        run for run in selected_runs if states[run.index] in {"ready", "launch_failed"}
    ]
    skipped_runs = [run for run in selected_runs if run not in eligible_runs]
    if skipped_runs:
        print("")
        print("Skipping runs that are already active or have execution artifacts:")
        for run in skipped_runs:
            print(f"  {run.index}. {run.name} ({states[run.index]})")
    if not eligible_runs:
        print("No selected runs are eligible to launch.")
        return []

    registry_path, lock_path = launcher_registry_paths(Path(plan.project_root))
    default_cap = registry_max_parallel(registry_path) or 1
    cap = max_parallel or prompt_positive_int(
        f"Maximum parallel experiments [{default_cap}]: ", default_cap
    )
    if cap < 1:
        raise SystemExit("--max-parallel must be a positive integer.")

    print("")
    print(f"Ready to launch: {len(eligible_runs)}")
    print(f"Maximum parallel experiments: {cap}")
    if selection is None and not prompt_yes_no("Proceed? [Y/n]: ", default=True):
        raise SystemExit("Launch cancelled.")

    launched = launch_runs(
        directory,
        plan,
        eligible_runs,
        max_parallel=cap,
        poll_seconds=poll_seconds,
        lost_grace_seconds=lost_grace_seconds,
        registry_path=registry_path,
        lock_path=lock_path,
        screen_executable=screen_executable,
    )
    print("")
    print(f"Launched {len(launched)} screen(s); the launcher is now exiting.")
    if launched:
        print(f"Attach with: screen -r {launched[0]}")
    return launched


def launch_runs(
    sweep_dir: Path,
    plan: SweepPlan,
    runs: list[SweepRun],
    *,
    max_parallel: int,
    poll_seconds: float,
    lost_grace_seconds: float,
    registry_path: Path,
    lock_path: Path,
    screen_executable: str,
) -> list[str]:
    launched: list[str] = []
    with launcher_lock(lock_path):
        sessions = list_screen_sessions(executable=screen_executable)
        registry = load_registry(registry_path)
        prepare_registry(registry, max_parallel=max_parallel, sessions=sessions)

        for run in runs:
            lane_index = choose_lane(registry, sessions)
            predecessor = active_tail(registry["lanes"][lane_index], sessions)
            session_name = screen_name(plan.sweep_id, run.name)
            if session_name in sessions:
                print(f"Skipping {run.name}: screen {session_name!r} already exists.")
                continue

            own_status_path = status_path(sweep_dir, run)
            own_log_path = log_path(sweep_dir, run)
            wait_for = predecessor_reference(predecessor) if predecessor else None
            status_payload = {
                "state": "launching",
                "sweep_id": plan.sweep_id,
                "run_name": run.name,
                "stage_run_id": run.stage_run_id,
                "screen_name": session_name,
                "lane": lane_index + 1,
                "wait_for": wait_for,
                "launched_at": utc_now(),
                "waiting_since": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
            }
            write_json_atomic(own_status_path, status_payload)

            tail = {
                "sweep_id": plan.sweep_id,
                "run_name": run.name,
                "status_path": str(own_status_path),
                "screen_name": session_name,
            }
            registry["lanes"][lane_index] = tail
            registry["next_lane"] = (lane_index + 1) % max_parallel
            save_registry(registry_path, registry)

            command = [
                sys.executable,
                "-m",
                "retrieval_core.sweeps.worker",
                "--sweep-dir",
                str(sweep_dir),
                "--run-name",
                run.name,
                "--poll-seconds",
                str(poll_seconds),
                "--lost-grace-seconds",
                str(lost_grace_seconds),
            ]
            try:
                launch_screen(
                    session_name=session_name,
                    log_file=own_log_path,
                    command=command,
                    cwd=Path(plan.project_root),
                    executable=screen_executable,
                )
            except BaseException as exc:
                status_payload.update(
                    state="launch_failed",
                    finished_at=utc_now(),
                    exit_code=1,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                write_json_atomic(own_status_path, status_payload)
                print(f"Could not launch {run.name}: {exc}")
                continue

            sessions.add(session_name)
            launched.append(session_name)
            dependency = predecessor["run_name"] if predecessor else "none"
            print(
                f"Launched {run.index}. {run.name} (lane {lane_index + 1}, waits for {dependency})"
            )
    return launched


def run_state(sweep_dir: Path, run: SweepRun, sessions: set[str]) -> str:
    if (Path(run.output_dir) / "manifest.json").is_file():
        return "succeeded"
    payload = read_status(status_path(sweep_dir, run))
    state = str(payload.get("state", ""))
    if state in ACTIVE_STATES:
        session_name = str(payload.get("screen_name", ""))
        return state if session_name in sessions else "lost"
    if state in TERMINAL_STATES:
        return state
    if Path(run.output_dir).exists():
        return "partial"
    return "ready"


def print_run_table(plan: SweepPlan, states: dict[int, str]) -> None:
    print(f"Sweep: {plan.sweep_id}")
    print("")
    width = max(len(STATE_LABELS.get(state, ("", state, ""))[1]) for state in states.values())
    use_color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    for run in plan.runs:
        state = states[run.index]
        symbol, label, color = STATE_LABELS.get(state, ("?", state.upper(), "red"))
        status_text = f"{symbol} {label:<{width}}"
        if use_color and color:
            status_text = f"{ANSI[color]}{status_text}{ANSI['reset']}"
        print(f"  {run.index:>3}. {status_text}  {run.name}")
    print("")


def parse_selection(
    selection: str,
    runs: list[SweepRun],
    states: dict[int, str],
) -> list[int]:
    normalized = selection.strip().lower()
    known_indices = {run.index for run in runs}
    if normalized == "all":
        return sorted(known_indices)
    if normalized == "ready":
        return sorted(
            index for index, state in states.items() if state in {"ready", "launch_failed"}
        )
    if not normalized:
        raise ValueError("Run selection cannot be empty.")

    selected: set[int] = set()
    for token in (item.strip() for item in normalized.split(",")):
        if not token:
            continue
        match = SELECTION_RANGE.fullmatch(token)
        if match:
            start, end = (int(value) for value in match.groups())
            if start > end:
                raise ValueError(f"Descending selection range is not supported: {token}")
            selected.update(range(start, end + 1))
            continue
        try:
            selected.add(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid run selection token: {token!r}") from exc

    unknown = selected - known_indices
    if unknown:
        raise ValueError(f"Run selection contains unknown indices: {sorted(unknown)}")
    return sorted(selected)


def choose_sweep_dir(sweep_dir: Path | None) -> Path:
    if sweep_dir is not None:
        resolved = sweep_dir.expanduser().resolve()
        if not (resolved / "sweep.yaml").is_file():
            raise FileNotFoundError(f"No sweep.yaml found in {resolved}")
        return resolved

    root = Path.cwd() / "artifacts" / "sweeps"
    choices = sorted(
        (path.parent for path in root.glob("*/sweep.yaml")),
        key=lambda path: path.name,
        reverse=True,
    )
    if not choices:
        raise FileNotFoundError(f"No prepared sweeps found below {root.resolve()}")
    print("Prepared sweeps:")
    for index, path in enumerate(choices, start=1):
        print(f"  {index}. {path.name}")
    while True:
        answer = input(f"Select 1-{len(choices)}: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= selected <= len(choices):
            return choices[selected - 1].resolve()
        print("Enter a number from the list.")


def launcher_registry_paths(project_root: Path) -> tuple[Path, Path]:
    directory = project_root / "artifacts" / "sweeps" / ".launcher"
    return directory / "lanes.json", directory / "lanes.lock"


def load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "max_parallel": None,
            "next_lane": 0,
            "lanes": [],
        }
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported launcher registry: {path}")
    return dict(payload)


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    write_json_atomic(path, registry)


def registry_max_parallel(path: Path) -> int | None:
    try:
        value = load_registry(path).get("max_parallel")
    except (FileNotFoundError, ValueError):
        return None
    return int(value) if value else None


def prepare_registry(registry: dict[str, Any], *, max_parallel: int, sessions: set[str]) -> None:
    lanes = list(registry.get("lanes", []))
    active = any(active_tail(tail, sessions) is not None for tail in lanes)
    existing_cap = registry.get("max_parallel")
    if active and existing_cap not in {None, max_parallel}:
        raise ValueError(
            f"The active launcher queue uses max_parallel={existing_cap}; "
            "the cap cannot change until every lane is terminal."
        )
    if not active or existing_cap is None:
        registry["max_parallel"] = max_parallel
        registry["next_lane"] = 0
        registry["lanes"] = [None] * max_parallel
        return
    if len(lanes) != max_parallel:
        raise ValueError("Launcher registry lane count does not match max_parallel.")


def choose_lane(registry: dict[str, Any], sessions: set[str]) -> int:
    lanes = registry["lanes"]
    start = int(registry.get("next_lane", 0)) % len(lanes)
    order = [(start + offset) % len(lanes) for offset in range(len(lanes))]
    for lane_index in order:
        if active_tail(lanes[lane_index], sessions) is None:
            return lane_index
    return start


def active_tail(tail: Any, sessions: set[str]) -> dict[str, Any] | None:
    if not isinstance(tail, dict):
        return None
    state = str(read_status(tail.get("status_path", "")).get("state", ""))
    if state in TERMINAL_STATES:
        return None
    if str(tail.get("screen_name", "")) not in sessions:
        return None
    return tail


def predecessor_reference(tail: dict[str, Any]) -> dict[str, str]:
    return {
        "sweep_id": str(tail["sweep_id"]),
        "run_name": str(tail["run_name"]),
        "status_path": str(tail["status_path"]),
        "screen_name": str(tail["screen_name"]),
    }


@contextmanager
def launcher_lock(path: Path) -> Iterator[None]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prompt_positive_int(prompt: str, default: int) -> int:
    while True:
        answer = input(prompt).strip()
        if not answer:
            return default
        try:
            parsed = int(answer)
        except ValueError:
            print("Enter a positive integer.")
            continue
        if parsed > 0:
            return parsed
        print("Enter a positive integer.")


def prompt_yes_no(prompt: str, *, default: bool) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


if __name__ == "__main__":
    main()
