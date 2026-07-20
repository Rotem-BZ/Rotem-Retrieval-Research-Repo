from pathlib import Path

from retrieval_core.utils.config import compose_stage_config, config_roots, core_config_dir


def test_bare_stage_name_resolves_to_stages_group() -> None:
    overrides = ["dataset=toy", "pipeline/inference@pipeline=dummy_keyword"]

    bare = compose_stage_config("inference", overrides)
    explicit = compose_stage_config("stages/inference", overrides)

    assert bare.stage.name == "inference"
    assert explicit.stage.name == bare.stage.name
    assert explicit.dataset == bare.dataset
    assert explicit.pipeline == bare.pipeline


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
    runs = experiment / "runs"
    configs.mkdir(parents=True)
    runs.mkdir()
    (configs / "inference.yaml").write_text(
        """defaults:
  - /stages/inference
  - override /dataset: toy
  - override /pipeline/inference@pipeline: dummy_keyword
  - _self_

runtime:
  device:
    device: cpu
""",
        encoding="utf-8",
    )
    (runs / "baseline.yaml").write_text(
        """# @package _global_
defaults:
  - /inference
  - _self_
""",
        encoding="utf-8",
    )

    cfg = compose_stage_config(
        "runs/baseline",
        experiment_dir=experiment,
    )

    assert cfg.stage.name == "inference"
    assert cfg.dataset.name == "toy"
    assert cfg.runtime.device.device == "cpu"
    assert cfg.stage.run_id == "example--baseline"
    assert cfg.experiment.run_name == "baseline"
    assert Path(cfg.paths.project_root) == project.resolve()
