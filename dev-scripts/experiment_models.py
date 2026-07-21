"""Experiment run definitions, discovery, and execution metadata."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from retrieval_core.stages import STAGE_RUNNERS
from retrieval_core.utils.config import compose_entrypoint_config
from retrieval_core.utils.io import read_json, write_json_atomic

TERMINAL_STATES = {"succeeded", "failed", "launch_failed", "cancelled", "lost"}
ACTIVE_STATES = {"launching", "waiting", "running"}


@dataclass(frozen=True)
class ExperimentRun:
    index: int
    name: str
    definition_file: Path
    stage_name: str
    stage_run_id: str
    output_dir: Path


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    name: str
    directory: Path
    project_root: Path
    runs: tuple[ExperimentRun, ...]


def load_plan(experiment_dir: str | Path) -> ExperimentPlan:
    """Discover and validate checked-in run definitions for one experiment."""

    directory = Path(experiment_dir).expanduser().resolve()
    project_root = project_root_for_experiment(directory)
    runs_dir = directory / "configs" / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Experiment runs directory does not exist: {runs_dir}")
    definition_files = sorted(runs_dir.glob("*.yaml"))
    if not definition_files:
        raise FileNotFoundError(f"No run config files found below {runs_dir}")

    experiment_id = slugify(directory.name, fallback="experiment")
    runs = tuple(
        load_run_definition(
            path,
            index=index,
            project_root=project_root,
        )
        for index, path in enumerate(definition_files, start=1)
    )
    return ExperimentPlan(
        experiment_id=experiment_id,
        name=directory.name,
        directory=directory,
        project_root=project_root,
        runs=runs,
    )


def load_run_definition(
    path: Path,
    *,
    index: int,
    project_root: Path,
) -> ExperimentRun:
    run_name = slugify(path.stem, fallback="run")
    if run_name != path.stem:
        raise ValueError(
            f"Run filenames must already be safe Hydra names; rename {path.name!r} "
            f"to {run_name + path.suffix!r}."
        )
    cfg = compose_entrypoint_config(path)
    stage_name = str(cfg.stage.name)
    if stage_name not in STAGE_RUNNERS:
        raise ValueError(f"Run {path} resolves to unknown stage {stage_name!r}.")
    output_dir = Path(str(OmegaConf.select(cfg, "stage.output_dir")))
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    return ExperimentRun(
        index=index,
        name=run_name,
        definition_file=path.resolve(),
        stage_name=stage_name,
        stage_run_id=str(cfg.stage.run_id),
        output_dir=output_dir.resolve(),
    )


def write_run_definition(
    path: Path,
    *,
    base_config: str,
    group_overrides: tuple[tuple[str, str], ...] = (),
    fields: dict[str, Any] | None = None,
) -> Path:
    """Write one minimal Hydra run config extending an experiment base config."""

    if path.exists():
        raise FileExistsError(f"Run definition already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_base = base_config.removesuffix(".yaml").strip("/")
    if not normalized_base:
        raise ValueError(
            "base_config must name a config below the experiment configs directory."
        )
    defaults: list[Any] = [f"/{normalized_base}"]
    defaults.extend(
        {f"override /{group.lstrip('/')}": choice} for group, choice in group_overrides
    )
    defaults.append("_self_")
    payload: dict[str, Any] = {"defaults": defaults}
    payload.update(fields or {})
    contents = "# @package _global_\n" + yaml.safe_dump(payload, sort_keys=False)
    path.write_text(contents, encoding="utf-8")
    return path


def render_hydra_command(run: ExperimentRun) -> str:
    tokens = [
        "uv",
        "run",
        "stage",
        run.stage_name,
        "--entrypoint",
        str(run.definition_file.resolve()),
    ]
    return shlex.join(tokens)


def experiment_state_dir(experiment_dir: str | Path, run: ExperimentRun | str) -> Path:
    directory = Path(experiment_dir).resolve()
    project_root = project_root_for_experiment(directory)
    name = run.name if isinstance(run, ExperimentRun) else str(run)
    return project_root / "artifacts" / "experiments" / directory.name / name


def status_path(experiment_dir: str | Path, run: ExperimentRun | str) -> Path:
    return experiment_state_dir(experiment_dir, run) / "status.json"


def read_status(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        return {}
    payload = read_json(resolved)
    return dict(payload) if isinstance(payload, dict) else {}


def update_status(path: str | Path, **changes: Any) -> dict[str, Any]:
    resolved = Path(path)
    payload = read_status(resolved)
    payload.update(changes)
    write_json_atomic(resolved, payload)
    return payload


def project_root_for_experiment(experiment_dir: Path) -> Path:
    if experiment_dir.parent.name != "experiments":
        raise ValueError(
            "Experiment directories must be located at <project>/experiments/<experiment>: "
            f"{experiment_dir}"
        )
    return experiment_dir.parent.parent.resolve()


def slugify(value: Any, *, fallback: str = "value") -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return text or fallback


def screen_name(experiment_id: str, run_name: str, *, max_length: int = 75) -> str:
    full_name = f"rr-{slugify(experiment_id)}--{slugify(run_name)}"
    if len(full_name) <= max_length:
        return full_name
    import hashlib

    digest = hashlib.sha256(full_name.encode("utf-8")).hexdigest()[:10]
    return f"{full_name[: max_length - len(digest) - 2].rstrip('-')}--{digest}"
