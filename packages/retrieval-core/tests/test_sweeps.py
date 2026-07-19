from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.command_builder import BuiltCommand
from retrieval_core.sweeps import launcher
from retrieval_core.sweeps.models import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentParameter,
    ExperimentPlan,
    ExperimentRun,
    choice_name,
    read_status,
    save_plan,
    status_path,
    update_status,
)
from retrieval_core.sweeps.prepare import (
    load_experiment_template,
    materialize_experiment,
    parameter_combinations,
    publish_experiment,
)
from retrieval_core.sweeps.worker import wait_for_predecessor
from retrieval_core.utils.artifacts import run_manifest

CONFIG_DIR = Path(__file__).parents[1] / "src" / "retrieval_core" / "configs"
REPOSITORY_ROOT = Path(__file__).parents[3]


def test_choice_name_describes_parameter_values() -> None:
    parameters = [
        ExperimentParameter("optimizer.learning_rate", "lr", [0.01]),
        ExperimentParameter("pipeline.chunk_size", "chunksize", [14]),
        ExperimentParameter("selection.model", "model", ["E5-base"]),
    ]

    assert choice_name(parameters, (0.01, 14, "E5-base")) == (
        "lr-0.01--chunksize-14--model-E5-base"
    )


def test_parameter_combinations_support_cartesian_and_zip() -> None:
    parameters = [
        ExperimentParameter("a", "a", [1, 2]),
        ExperimentParameter("b", "b", ["x", "y"]),
    ]

    assert parameter_combinations(parameters, "cartesian") == [
        (1, "x"),
        (1, "y"),
        (2, "x"),
        (2, "y"),
    ]
    assert parameter_combinations(parameters, "zip") == [(1, "x"), (2, "y")]
    assert parameter_combinations([], "single") == [()]


def test_load_experiment_template(tmp_path: Path) -> None:
    template = tmp_path / "configs" / "matrix.yaml"
    template.parent.mkdir()
    template.write_text(
        """schema_version: 1
stage: inference
base_overrides:
  - dataset=toy
combination_mode: zip
parameters:
  - path: pipeline/inference@pipeline
    label: variant
    values: [dense_jsonl, treatment]
    raw: true
""",
        encoding="utf-8",
    )

    built, parameters, combination_mode = load_experiment_template(template)

    assert built.stage_name == "inference"
    assert built.overrides == ("dataset=toy",)
    assert parameters == [
        ExperimentParameter(
            "pipeline/inference@pipeline",
            "variant",
            ["dense_jsonl", "treatment"],
            raw=True,
        )
    ]
    assert combination_mode == "zip"


def test_publish_experiment_preserves_research_files_and_template(tmp_path: Path) -> None:
    destination = tmp_path / "experiments" / "example"
    (destination / "configs").mkdir(parents=True)
    (destination / "experiment.md").write_text("# Research plan\n", encoding="utf-8")
    (destination / "configs" / "matrix.yaml").write_text("stage: inference\n")
    staging = tmp_path / "experiments" / ".example.tmp"
    (staging / "runs" / "baseline").mkdir(parents=True)
    (staging / "experiment.yaml").write_text("experiment_id: example\n")
    (staging / "runs" / "baseline" / "config.yaml").write_text("stage: {}\n")

    publish_experiment(staging, destination)

    assert (destination / "experiment.md").read_text(encoding="utf-8") == "# Research plan\n"
    assert (destination / "configs" / "matrix.yaml").read_text() == "stage: inference\n"
    assert (destination / "runs" / "baseline" / "config.yaml").is_file()
    assert (destination / "experiment.yaml").is_file()


def test_parse_selection_supports_numbers_ranges_and_status_aliases(tmp_path: Path) -> None:
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
    first = tmp_path / "experiments" / "first"
    second = tmp_path / "experiments" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    save_plan(first / "experiment.yaml", _plan(tmp_path, run_count=1))
    save_plan(second / "experiment.yaml", _plan(tmp_path, run_count=2))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert launcher.choose_experiment_dir(None) == second.resolve()
    assert launcher.choose_experiment_dir(None, experiment_name="first") == first.resolve()


def test_launch_runs_builds_dependency_lanes_and_returns_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path, run_count=5)
    experiment_dir = tmp_path / "experiments" / plan.experiment_id
    experiment_dir.mkdir(parents=True)
    registry_path, lock_path = launcher.launcher_registry_paths(tmp_path)
    launched_commands: list[list[str]] = []

    monkeypatch.setattr(launcher, "launcher_lock", lambda _path: nullcontext())
    monkeypatch.setattr(launcher, "list_screen_sessions", lambda **_kwargs: set())

    def fake_launch_screen(**kwargs) -> None:
        launched_commands.append(list(kwargs["command"]))

    monkeypatch.setattr(launcher, "launch_screen", fake_launch_screen)

    launched = launcher.launch_runs(
        experiment_dir,
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
    statuses = [read_status(status_path(experiment_dir, run)) for run in plan.runs]
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
                "experiment_id": "experiment",
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


def test_materialize_experiment_writes_resolved_config_inside_run_dir(tmp_path: Path) -> None:
    parameter = ExperimentParameter(
        "pipeline.components.indexer.init_parameters.overwrite",
        "overwrite",
        [True],
    )
    built = BuiltCommand(
        stage_name="indexing",
        overrides=("dataset=toy", "pipeline/indexing@pipeline=dummy_jsonl"),
        command="unused",
    )

    plan = materialize_experiment(
        tmp_path,
        experiment_id="test-experiment",
        experiment_name="test-experiment",
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
    assert plan.runs[0].config_file == "runs/overwrite-true/config.yaml"
    assert "experiment:" in config_text
    assert "id: test-experiment" in config_text
    assert "preserve_run_config: true" in config_text
    assert "${" not in config_text
    assert "???" not in config_text


def test_run_manifest_links_stage_artifact_to_experiment(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"project_root": str(tmp_path), "runs_dir": "artifacts/runs"},
            "stage": {"name": "inference", "run_id": "experiment--baseline"},
            "experiment": {
                "id": "experiment",
                "name": "experiment",
                "run_name": "baseline",
                "parameters": {"pipeline": "dense_jsonl"},
            },
        }
    )

    manifest = run_manifest(cfg, artifacts={"predictions": "predictions.json"})

    assert manifest["experiment"] == {
        "id": "experiment",
        "name": "experiment",
        "run_name": "baseline",
        "parameters": {"pipeline": "dense_jsonl"},
    }


def _run(index: int, root: Path) -> ExperimentRun:
    name = f"value-{index}"
    return ExperimentRun(
        index=index,
        name=name,
        stage_run_id=f"experiment--{name}",
        config_file=f"runs/{name}/config.yaml",
        config_sha256="checksum",
        parameters={"value": index},
        output_dir=str(root / "outputs" / name),
    )


def _plan(root: Path, *, run_count: int) -> ExperimentPlan:
    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id="test-experiment",
        name="test-experiment",
        stage="inference",
        created_at="2026-01-01T00:00:00+00:00",
        project_root=str(root),
        source_config_dir=str(root / "configs"),
        combination_mode="cartesian",
        parameters=[ExperimentParameter("value", "value", list(range(1, run_count + 1)))],
        runs=[_run(index, root) for index in range(1, run_count + 1)],
    )
