from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import interactive_run_in_parallel_screens as launcher
from interactive_create_run import split_run_overrides
from _internal.experiment_models import (
    ExperimentRun,
    load_plan,
    read_status,
    render_hydra_command,
    status_path,
    update_status,
    write_run_definition,
)
from _internal.experiment_worker import wait_for_predecessor
from retrieval_core.utils.artifacts import run_manifest


def test_run_definitions_are_minimal_hydra_configs_and_compose(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    definition = write_run_definition(
        experiment / "configs" / "runs" / "baseline.yaml",
        base_config="base-experiment-configs/indexing",
    )

    plan = load_plan(experiment)
    run = plan.runs[0]

    assert definition.read_text(encoding="utf-8") == (
        "# @package _global_\ndefaults:\n"
        "- /base-experiment-configs/indexing\n- _self_\n"
    )
    assert run.name == "baseline"
    assert run.stage_name == "indexing"
    assert run.stage_run_id == "example--baseline"
    assert (
        run.output_dir
        == (
            tmp_path / "artifacts" / "runs" / "indexing" / "example--baseline"
        ).resolve()
    )
    command = render_hydra_command(run)
    assert "stage indexing" in command
    assert "--entrypoint" in command
    assert "baseline.yaml" in command
    assert "dataset=" not in command


def test_run_definition_can_override_only_changed_fields(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    write_run_definition(
        experiment / "configs" / "runs" / "smaller.yaml",
        base_config="base-experiment-configs/indexing",
        fields={"runtime": {"concurrency_limit": 2}},
    )

    run = load_plan(experiment).runs[0]
    assert run.name == "smaller"
    assert run.stage_run_id == "example--smaller"


def test_create_run_splits_group_selections_from_value_fields(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    groups, fields = split_run_overrides(
        (
            "dataset=toy",
            "pipeline/indexing@pipeline=scaffold/documents_jsonl",
            "runtime=cpu",
            "runtime.concurrency_limit=2",
        ),
        config_dir=experiment / "configs",
    )

    assert groups == (
        ("dataset", "toy"),
        ("pipeline/indexing@pipeline", "scaffold/documents_jsonl"),
        ("runtime", "cpu"),
    )
    assert fields == {"runtime": {"concurrency_limit": 2}}

    with pytest.raises(ValueError, match="launcher-controlled"):
        split_run_overrides(
            ("stage.run_id=manual",),
            config_dir=experiment / "configs",
        )


def test_parse_selection_supports_numbers_ranges_and_status_aliases(
    tmp_path: Path,
) -> None:
    runs = [_run(index, tmp_path) for index in range(1, 8)]
    states = {index: "ready" for index in range(1, 8)}
    states[2] = "succeeded"
    states[6] = "failed"

    assert launcher.parse_selection("1, 3, 4-7", runs, states) == [1, 3, 4, 5, 6, 7]
    assert launcher.parse_selection("ready", runs, states) == [1, 3, 4, 5, 7]

    with pytest.raises(ValueError, match="unknown indices"):
        launcher.parse_selection("8", runs, states)


def test_choose_experiment_from_project_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _experiment(tmp_path, name="first")
    second = _experiment(tmp_path, name="second")
    _write_indexing_run(first, "one")
    _write_indexing_run(second, "one")
    _write_indexing_run(second, "two")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert launcher.choose_experiment_dir(None) == second.resolve()
    assert (
        launcher.choose_experiment_dir(None, experiment_name="first") == first.resolve()
    )


def test_launch_runs_builds_dependency_lanes_and_writes_state_under_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = _experiment(tmp_path)
    for index in range(1, 6):
        _write_indexing_run(experiment, f"value-{index}")
    plan = load_plan(experiment)
    registry_path, lock_path = launcher.launcher_registry_paths(tmp_path)
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(launcher, "launcher_lock", lambda _path: nullcontext())
    monkeypatch.setattr(launcher, "list_screen_sessions", lambda **_kwargs: set())
    monkeypatch.setattr(
        launcher,
        "launch_screen",
        lambda **kwargs: launched_commands.append(list(kwargs["command"])),
    )

    launched = launcher.launch_runs(
        experiment,
        plan,
        list(plan.runs),
        max_parallel=2,
        poll_seconds=0.1,
        lost_grace_seconds=1,
        registry_path=registry_path,
        lock_path=lock_path,
        screen_executable="screen",
    )

    assert len(launched) == 5
    assert len(launched_commands) == 5
    assert all("--entrypoint" in command for command in launched_commands)
    assert all("--experiment-dir" not in command for command in launched_commands)
    statuses = [read_status(status_path(experiment, run)) for run in plan.runs]
    assert [status["lane"] for status in statuses] == [1, 2, 1, 2, 1]
    assert statuses[2]["wait_for"]["run_name"] == plan.runs[0].name
    assert status_path(experiment, plan.runs[0]).parent == (
        tmp_path / "artifacts" / "experiments" / "example" / "value-1"
    )


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
                "experiment_id": "experiment",
            }
        ],
    }

    with pytest.raises(ValueError, match="cannot change"):
        launcher.prepare_registry(registry, max_parallel=2, sessions={"rr-active"})


def test_waiting_worker_releases_lane_on_terminal_or_lost_predecessor(
    tmp_path: Path,
) -> None:
    predecessor_status = tmp_path / "predecessor.json"
    predecessor = {
        "status_path": str(predecessor_status),
        "screen_name": "rr-predecessor",
    }
    update_status(predecessor_status, state="failed")
    assert (
        wait_for_predecessor(
            predecessor,
            poll_seconds=0,
            lost_grace_seconds=5,
        )
        == "failed"
    )

    update_status(predecessor_status, state="running")
    times = iter([0.0, 5.0])
    assert (
        wait_for_predecessor(
            predecessor,
            poll_seconds=0,
            lost_grace_seconds=5,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: next(times),
            session_exists_fn=lambda _name: False,
        )
        == "lost"
    )


def test_run_manifest_links_stage_artifact_to_experiment(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"project_root": str(tmp_path), "runs_dir": "artifacts/runs"},
            "stage": {"name": "inference", "run_id": "experiment--baseline"},
            "experiment": {
                "id": "experiment",
                "name": "experiment",
                "run_name": "baseline",
                "parameters": {},
            },
        }
    )

    manifest = run_manifest(cfg, artifacts={"predictions": "predictions.json"})

    assert manifest["experiment"] == {
        "id": "experiment",
        "name": "experiment",
        "run_name": "baseline",
        "parameters": {},
    }


def _experiment(root: Path, *, name: str = "example") -> Path:
    experiment = root / "experiments" / name
    configs = experiment / "configs"
    (configs / "base-experiment-configs").mkdir(parents=True)
    (configs / "runs").mkdir()
    (configs / "base-experiment-configs" / "indexing.yaml").write_text(
        """# @package _global_
defaults:
  - /stages/indexing
  - override /dataset: toy
  - override /pipeline/indexing@pipeline: scaffold/documents_jsonl
  - override /runtime: cpu
  - _self_

selections:
  index_id: test-index
""",
        encoding="utf-8",
    )
    return experiment


def _write_indexing_run(experiment: Path, name: str) -> Path:
    return write_run_definition(
        experiment / "configs" / "runs" / f"{name}.yaml",
        base_config="base-experiment-configs/indexing",
    )


def _run(index: int, root: Path) -> ExperimentRun:
    name = f"value-{index}"
    return ExperimentRun(
        index=index,
        name=name,
        definition_file=root / f"{name}.yaml",
        stage_name="indexing",
        stage_run_id=f"experiment--{name}",
        output_dir=root / "outputs" / name,
    )
