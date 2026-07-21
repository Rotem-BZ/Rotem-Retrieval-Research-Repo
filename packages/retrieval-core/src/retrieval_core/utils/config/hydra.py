"""Hydra configuration loading for project and core config directories."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import re
from typing import Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from retrieval_core.utils.config.searchpath import use_config_fallbacks


def compose_stage_config(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> DictConfig:
    """Compose with experiment, project, then core config precedence."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    roots = list(
        config_roots(
            config_dir,
            experiment_dir=experiment_dir,
            project_dir=project_dir,
        )
    )
    experiment = (
        _resolve_optional_directory(experiment_dir, label="Experiment")
        if experiment_dir is not None
        else None
    )
    resolved_config_name = _resolve_entry_config_name(config_name, roots)
    run_name = _experiment_run_name(resolved_config_name)
    if run_name is not None:
        if experiment is None:
            raise ValueError("Run configs require experiment_dir.")
        run_file = experiment / "runs" / f"{run_name}.yaml"
        if not run_file.is_file():
            raise FileNotFoundError(f"Experiment run config does not exist: {run_file}")
        roots.insert(0, experiment)
    fallbacks = [
        (f"retrieval-config-{index}", _config_uri(path))
        for index, path in enumerate(roots[1:], start=1)
    ]
    with use_config_fallbacks(fallbacks):
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(roots[0]),
            job_name=f"stage-{resolved_config_name}",
        ):
            cfg = compose(
                config_name=resolved_config_name,
                overrides=list(overrides or []),
            )

    if run_name is not None:
        assert experiment is not None
        _apply_experiment_run_identity(cfg, experiment=experiment, run_name=run_name)
    return cfg


def core_config_dir() -> Path:
    """Return the config directory shipped by ``retrieval-core``."""

    return Path(str(files("retrieval_core.configs")))


def find_config_dir(
    config_dir: str | Path | None = None,
    *,
    experiment_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> Path:
    """Resolve the first config root in the active precedence chain."""

    return config_roots(
        config_dir,
        experiment_dir=experiment_dir,
        project_dir=project_dir,
    )[0]


def config_roots(
    config_dir: str | Path | None = None,
    *,
    experiment_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return experiment, project, and core roots in Hydra precedence order."""

    if config_dir is not None and experiment_dir is not None:
        raise ValueError("Pass either config_dir or experiment_dir, not both.")

    roots: list[Path] = []
    resolved_project = _resolve_optional_directory(project_dir, label="Project")

    if experiment_dir is not None:
        experiment = _resolve_optional_directory(experiment_dir, label="Experiment")
        assert experiment is not None
        experiment_configs = experiment / "configs"
        if experiment_configs.is_dir():
            roots.append(experiment_configs.resolve())
        inferred_project = _project_root_for_experiment(experiment)
        if resolved_project is not None and resolved_project != inferred_project:
            raise ValueError(f"Experiment {experiment} is not inside project {resolved_project}.")
        resolved_project = inferred_project
    elif config_dir is not None:
        primary = _resolve_optional_directory(config_dir, label="Config")
        assert primary is not None
        roots.append(primary)
        resolved_project = resolved_project or _infer_project_root_from_config_dir(primary)

    if resolved_project is None:
        candidate = Path.cwd().resolve()
        if (candidate / "configs").is_dir():
            resolved_project = candidate

    if resolved_project is not None:
        project_configs = (resolved_project / "configs").resolve()
        if project_configs.is_dir():
            roots.append(project_configs)

    core = core_config_dir().resolve()
    roots.append(core)
    return tuple(dict.fromkeys(roots))


def _resolve_optional_directory(
    path: str | Path | None,
    *,
    label: str,
) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def _project_root_for_experiment(experiment_dir: Path) -> Path:
    if experiment_dir.parent.name != "experiments":
        raise ValueError(
            "Experiment directories must be located at <project>/experiments/<experiment>: "
            f"{experiment_dir}"
        )
    return experiment_dir.parent.parent.resolve()


def _infer_project_root_from_config_dir(config_dir: Path) -> Path | None:
    if config_dir.name != "configs":
        return None
    if config_dir.parent.parent.name == "experiments":
        return config_dir.parent.parent.parent.resolve()
    return config_dir.parent.resolve()


def _config_uri(path: Path) -> str:
    core = core_config_dir().resolve()
    if path == core:
        return "pkg://retrieval_core.configs"
    return f"file://{path.as_posix()}"


def _experiment_run_name(config_name: str) -> str | None:
    normalized = config_name.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    if parts[0] != "runs":
        return None
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Experiment run configs must be named runs/<run-name>.")
    return parts[1]


def _resolve_entry_config_name(config_name: str, roots: Sequence[Path]) -> str:
    """Resolve bare stage names while preserving explicit top-level configs."""

    normalized = config_name.replace("\\", "/").removesuffix(".yaml").strip("/")
    if "/" in normalized:
        return normalized
    if any((root / f"{normalized}.yaml").is_file() for root in roots):
        return normalized
    stage_name = f"stages/{normalized}"
    if any((root / f"{stage_name}.yaml").is_file() for root in roots):
        return stage_name
    return normalized


def _apply_experiment_run_identity(
    cfg: DictConfig,
    *,
    experiment: Path,
    run_name: str,
) -> None:
    project_root = _project_root_for_experiment(experiment)
    experiment_id = _slugify(experiment.name, fallback="experiment")
    normalized_run_name = _slugify(run_name, fallback="run")
    parameters = OmegaConf.select(cfg, "experiment.parameters", default={})
    OmegaConf.update(
        cfg,
        "paths.project_root",
        project_root.as_posix(),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "stage.run_id",
        f"{experiment_id}--{normalized_run_name}",
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "experiment",
        {
            "id": experiment_id,
            "name": experiment.name,
            "run_name": normalized_run_name,
            "parameters": parameters,
        },
        merge=False,
        force_add=True,
    )


def _slugify(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return text or fallback
