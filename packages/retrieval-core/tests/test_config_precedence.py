import re
from pathlib import Path

import pytest

from retrieval_core.utils.config import (
    compose_entrypoint_config,
    compose_stage_config,
    config_roots,
    core_config_dir,
    resolve_config_entrypoint,
)


def test_bare_stage_name_resolves_to_stages_group() -> None:
    overrides = [
        "dataset=toy",
        "pipeline/inference@pipeline=retrieve/dense_in_memory",
        "selections/embedding_model=e5/small_v2",
        "runtime=gpu",
    ]

    bare = compose_stage_config("inference", overrides)
    explicit = compose_stage_config("stages/inference", overrides)

    assert bare.stage.name == "inference"
    assert explicit.stage.name == bare.stage.name
    assert explicit.dataset == bare.dataset
    assert explicit.pipeline == bare.pipeline


def test_non_experiment_stage_keeps_timestamp_default_and_allows_override() -> None:
    generated = compose_stage_config("evaluation", ["dataset=toy"])
    explicit = compose_stage_config(
        "evaluation",
        ["dataset=toy", "stage.run_id=manual-evaluation"],
    )

    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}", str(generated.stage.run_id))
    assert explicit.stage.run_id == "manual-evaluation"


def test_paths_expose_git_root_and_use_its_data_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repository"
    project = repository / "projects" / "example"
    repository_data = repository / "data"
    repository_data.mkdir(parents=True)
    project.mkdir(parents=True)
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "retrieval_core.utils.config.hydra.find_git_root",
        lambda: repository.resolve(),
    )

    cfg = compose_stage_config("evaluation", ["dataset=toy"])

    assert Path(cfg.paths.repo_root) == repository.resolve()
    assert Path(cfg.paths.data_dir) == repository_data.resolve()
    assert Path(cfg.paths.processed_data_dir) == repository_data.resolve() / "processed"
    assert Path(cfg.paths.artifacts_dir) == Path("artifacts")


def test_experiment_configs_override_project_then_core(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiment = project / "experiments" / "example"
    project_dataset = project / "configs" / "dataset" / "shared.yaml"
    experiment_dataset = experiment / "configs" / "dataset" / "shared.yaml"
    project_dataset.parent.mkdir(parents=True)
    experiment_dataset.parent.mkdir(parents=True)
    project_dataset.write_text("name: project\nqrels_path: project-qrels.jsonl\n", encoding="utf-8")
    experiment_dataset.write_text(
        "name: experiment\nqrels_path: experiment-qrels.jsonl\n",
        encoding="utf-8",
    )

    experiment_cfg = compose_stage_config(
        "evaluation",
        ["dataset=shared"],
        experiment_dir=experiment,
    )

    assert experiment_cfg.dataset.name == "experiment"
    assert config_roots(experiment_dir=experiment) == (
        experiment.resolve() / "configs",
        project.resolve() / "configs",
        core_config_dir().resolve(),
    )

    experiment_dataset.unlink()
    project_cfg = compose_stage_config(
        "evaluation",
        ["dataset=shared"],
        experiment_dir=experiment,
    )
    core_cfg = compose_stage_config(
        "evaluation",
        ["dataset=toy"],
        experiment_dir=experiment,
    )

    assert project_cfg.dataset.name == "project"
    assert core_cfg.dataset.name == "toy"


def test_experiment_directory_must_follow_project_layout(tmp_path: Path) -> None:
    experiment = tmp_path / "standalone"
    (experiment / "configs").mkdir(parents=True)

    try:
        config_roots(experiment_dir=experiment)
    except ValueError as exc:
        assert "<project>/experiments/<experiment>" in str(exc)
    else:
        raise AssertionError("Expected invalid experiment layout to be rejected")


def test_experiment_configs_are_optional(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiment = project / "experiments" / "example"
    experiment.mkdir(parents=True)

    cfg = compose_stage_config(
        "evaluation",
        ["dataset=toy"],
        experiment_dir=experiment,
    )

    assert cfg.dataset.name == "toy"
    assert config_roots(experiment_dir=experiment) == (core_config_dir().resolve(),)


def test_run_config_uses_hydra_defaults_without_cli_overrides(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiment = project / "experiments" / "example"
    configs = experiment / "configs"
    base_configs = configs / "base-experiment-configs"
    runs = configs / "runs"
    base_configs.mkdir(parents=True)
    runs.mkdir()
    (base_configs / "inference.yaml").write_text(
        """# @package _global_
defaults:
  - /stages/inference
  - override /dataset: toy
  - override /pipeline/inference@pipeline: retrieve/dense_in_memory
  - override /selections/embedding_model@selections.embedding_model: e5/small_v2
  - override /runtime: cpu
  - _self_

selections:
  index_id: test-index
""",
        encoding="utf-8",
    )
    (runs / "baseline.yaml").write_text(
        """# @package _global_
defaults:
  - /base-experiment-configs/inference
  - _self_

stage:
  run_id: explicit-baseline
""",
        encoding="utf-8",
    )

    entrypoint = runs / "baseline.yaml"
    resolved = resolve_config_entrypoint(entrypoint)
    cfg = compose_entrypoint_config(entrypoint)

    assert resolved.config_dir == configs.resolve()
    assert resolved.config_name == "runs/baseline"
    assert resolved.experiment_dir == experiment.resolve()
    assert resolved.project_dir == project.resolve()
    assert cfg.stage.name == "inference"
    assert cfg.dataset.name == "toy"
    assert cfg.runtime.device.device == "cpu"
    assert cfg.stage.run_id == "explicit-baseline"
    assert cfg.experiment.run_name == "baseline"
    assert Path(cfg.paths.project_root) == project.resolve()

    with pytest.raises(ValueError, match="may not be overridden"):
        compose_entrypoint_config(entrypoint, ["stage.run_id=other-id"])


def test_experiment_run_config_requires_direct_stage_run_id(tmp_path: Path) -> None:
    run_file = (
        tmp_path
        / "project"
        / "experiments"
        / "example"
        / "configs"
        / "runs"
        / "missing.yaml"
    )
    run_file.parent.mkdir(parents=True)
    run_file.write_text("defaults:\n  - _self_\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must define stage.run_id directly"):
        compose_entrypoint_config(run_file)


def test_entrypoint_must_be_yaml_below_configs(tmp_path: Path) -> None:
    outside = tmp_path / "entrypoint.yaml"
    outside.write_text("value: true\n", encoding="utf-8")

    try:
        resolve_config_entrypoint(outside)
    except ValueError as exc:
        assert "below a configs/ directory" in str(exc)
    else:
        raise AssertionError("Expected an entrypoint outside configs/ to be rejected")
