from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from retrieval_core.command_builder import BuiltCommand
from retrieval_core.sweeps import launcher
from retrieval_core.sweeps.models import (
    SWEEP_SCHEMA_VERSION,
    SweepParameter,
    SweepPlan,
    SweepRun,
    choice_name,
    read_status,
    status_path,
    update_status,
)
from retrieval_core.sweeps.prepare import materialize_sweep, parameter_combinations
from retrieval_core.sweeps.worker import wait_for_predecessor

CONFIG_DIR = Path(__file__).parents[1] / "src" / "retrieval_core" / "configs"
REPOSITORY_ROOT = Path(__file__).parents[3]


def test_choice_name_describes_parameter_values() -> None:
    parameters = [
        SweepParameter("optimizer.learning_rate", "lr", [0.01]),
        SweepParameter("pipeline.chunk_size", "chunksize", [14]),
        SweepParameter("selection.model", "model", ["E5-base"]),
    ]

    assert choice_name(parameters, (0.01, 14, "E5-base")) == (
        "lr-0.01--chunksize-14--model-E5-base"
    )


def test_parameter_combinations_support_cartesian_and_zip() -> None:
    parameters = [
        SweepParameter("a", "a", [1, 2]),
        SweepParameter("b", "b", ["x", "y"]),
    ]

    assert parameter_combinations(parameters, "cartesian") == [
        (1, "x"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
    ]
    assert parameter_combinations(parameters, "zip") == [(1, "x"), (2, "y")]


def test_parse_selection_supports_numbers_ranges_and_status_aliases(tmp_path: Path) -> None:
    runs = [_run(index, tmp_path) for index in range(1, 8)]
    states = {index: "ready" for index in range(1, 8)}
    states[2] = "succeeded"
    states[6] = "failed"

    assert launcher.parse_selection("1, 3, 4-7", runs, states) == [1, 3, 4, 5, 6, 7]
    assert launcher.parse_selection("ready", runs, states) == [1, 3, 4, 5, 7]

    with pytest.raises(ValueError, match="unknown indices"):
        launcher.parse_selection("8", runs, states)


def test_launch_runs_builds_dependency_lanes_and_returns_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, run_count=5)
    sweep_dir = tmp_path / "artifacts" / "sweeps" / plan.sweep_id
    sweep_dir.mkdir(parents=True)
    registry_path, lock_path = launcher.launcher_registry_paths(tmp_path)
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(launcher, "launcher_lock", lambda _path: nullcontext())
    monkeypatch.setattr(launcher, "list_screen_sessions", lambda **_kwargs: set())

    def fake_launch_screen(**kwargs) -> None:
        launched_commands.append(list(kwargs["command"]))

    monkeypatch.setattr(launcher, "launch_screen", fake_launch_screen)

    launched = launcher.launch_runs(
        sweep_dir,
        plan,
        plan.runs,
        max_parallel=2,
        poll_seconds=0.1,
        lost_grace_seconds=1,
        registry_path=registry_path,
        lock_path=lock_path,
        screen_executable="screen",
    )

    assert len(launched) == 5
    assert len(launched_commands) == 5
    statuses = [read_status(status_path(sweep_dir, run)) for run in plan.runs]
    assert [status["lane"] for status in statuses] == [1, 2, 1, 2, 1]
    assert statuses[0]["wait_for"] is None
    assert statuses[1]["wait_for"] is None
    assert statuses[2]["wait_for"]["run_name"] == plan.runs[0].name
    assert statuses[3]["wait_for"]["run_name"] == plan.runs[1].name
    assert statuses[4]["wait_for"]["run_name"] == plan.runs[2].name


def test_active_lane_cap_cannot_change(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    update_status(status, state="running")
    registry = {
        "schema_version": 1,
        "max_parallel": 1,
        "next_lane": 0,
        "lanes": [
            {
                "status_path": str(status),
                "screen_name": "rr-active",
                "run_name": "active",
                "sweep_id": "sweep",
            }
        ],
    }

    with pytest.raises(ValueError, match="cannot change"):
        launcher.prepare_registry(registry, max_parallel=2, sessions={"rr-active"})


def test_waiting_worker_releases_lane_on_terminal_or_lost_predecessor(tmp_path: Path) -> None:
    predecessor_status = tmp_path / "predecessor.json"
    predecessor = {
        "status_path": str(predecessor_status),
        "screen_name": "rr-predecessor",
    }
    update_status(predecessor_status, state="failed")
    assert wait_for_predecessor(
        predecessor,
        poll_seconds=0,
        lost_grace_seconds=5,
    ) == "failed"

    update_status(predecessor_status, state="running")
    times = iter([0.0, 5.0])
    assert wait_for_predecessor(
        predecessor,
        poll_seconds=0,
        lost_grace_seconds=5,
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: next(times),
        session_exists_fn=lambda _name: False,
    ) == "lost"


def test_materialize_sweep_writes_fully_resolved_immutable_config(tmp_path: Path) -> None:
    parameter = SweepParameter(
        "pipeline.components.indexer.init_parameters.overwrite",
        "overwrite",
        [True],
    )
    built = BuiltCommand(
        stage_name="indexing",
        overrides=("dataset=toy", "pipeline/indexing@pipeline=dummy_jsonl"),
        command="unused",
    )

    plan = materialize_sweep(
        tmp_path,
        sweep_id="test-sweep--20260101-000000",
        sweep_name="test-sweep",
        built=built,
        parameters=[parameter],
        combinations=[(True,)],
        combination_mode="cartesian",
        project_root=REPOSITORY_ROOT,
        config_dir=CONFIG_DIR,
        output_fn=lambda _message: None,
    )

    config_text = (tmp_path / plan.runs[0].config_file).read_text(encoding="utf-8")
    assert plan.runs[0].name == "overwrite-true"
    assert "preserve_run_config: true" in config_text
    assert "${" not in config_text
    assert "???" not in config_text


def _run(index: int, root: Path) -> SweepRun:
    name = f"value-{index}"
    return SweepRun(
        index=index,
        name=name,
        stage_run_id=f"sweep--{name}",
        config_file=f"configs/{name}.yaml",
        config_sha256="checksum",
        parameters={"value": index},
        output_dir=str(root / "outputs" / name),
    )


def _plan(root: Path, *, run_count: int) -> SweepPlan:
    return SweepPlan(
        schema_version=SWEEP_SCHEMA_VERSION,
        sweep_id="test-sweep--20260101-000000",
        name="test-sweep",
        stage="inference",
        created_at="2026-01-01T00:00:00+00:00",
        project_root=str(root),
        source_config_dir=str(root / "configs"),
        combination_mode="cartesian",
        parameters=[SweepParameter("value", "value", list(range(1, run_count + 1)))],
        runs=[_run(index, root) for index in range(1, run_count + 1)],
    )
