"""Persistent models and naming helpers for prepared experiments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from retrieval_core.utils.io import read_json, write_json_atomic
from retrieval_core.utils.hashing import sha256_text

EXPERIMENT_SCHEMA_VERSION = 1
TERMINAL_STATES = {"succeeded", "failed", "launch_failed", "cancelled", "lost"}
ACTIVE_STATES = {"launching", "waiting", "running"}


@dataclass(frozen=True)
class ExperimentParameter:
    path: str
    label: str
    values: list[Any]
    raw: bool = False


@dataclass(frozen=True)
class ExperimentRun:
    index: int
    name: str
    stage_run_id: str
    config_file: str
    config_sha256: str
    parameters: dict[str, Any]
    output_dir: str


@dataclass(frozen=True)
class ExperimentPlan:
    schema_version: int
    experiment_id: str
    name: str
    stage: str
    created_at: str
    project_root: str
    source_config_dir: str
    combination_mode: str
    parameters: list[ExperimentParameter]
    runs: list[ExperimentRun]


def save_plan(path: Path, plan: ExperimentPlan) -> None:
    payload = asdict(plan)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_plan(experiment_dir: str | Path) -> ExperimentPlan:
    """Load a prepared experiment, accepting legacy ``sweep.yaml`` plans."""

    directory = Path(experiment_dir).expanduser().resolve()
    plan_path = directory / "experiment.yaml"
    if not plan_path.is_file():
        plan_path = directory / "sweep.yaml"
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment schema in {plan_path}.")
    experiment_id = payload.get("experiment_id", payload.get("sweep_id"))
    if not experiment_id:
        raise ValueError(f"Experiment plan has no experiment_id: {plan_path}")
    return ExperimentPlan(
        schema_version=int(payload["schema_version"]),
        experiment_id=str(experiment_id),
        name=str(payload["name"]),
        stage=str(payload["stage"]),
        created_at=str(payload["created_at"]),
        project_root=str(payload["project_root"]),
        source_config_dir=str(payload["source_config_dir"]),
        combination_mode=str(payload["combination_mode"]),
        parameters=[ExperimentParameter(**item) for item in payload.get("parameters", [])],
        runs=[ExperimentRun(**item) for item in payload.get("runs", [])],
    )


def run_by_name(plan: ExperimentPlan, name: str) -> ExperimentRun:
    for run in plan.runs:
        if run.name == name:
            return run
    raise KeyError(f"Experiment {plan.experiment_id!r} has no run named {name!r}.")


def status_path(experiment_dir: str | Path, run: ExperimentRun | str) -> Path:
    name = run.name if isinstance(run, ExperimentRun) else run
    return Path(experiment_dir).resolve() / "runs" / name / "status.json"


def log_path(experiment_dir: str | Path, run: ExperimentRun | str) -> Path:
    name = run.name if isinstance(run, ExperimentRun) else run
    return Path(experiment_dir).resolve() / "runs" / name / "screen.log"


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


def is_terminal_status(path: str | Path) -> bool:
    return str(read_status(path).get("state", "")) in TERMINAL_STATES


def slugify(value: Any, *, fallback: str = "value") -> str:
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    return text or fallback


def choice_name(
    parameters: list[ExperimentParameter],
    values: tuple[Any, ...],
    *,
    max_length: int = 120,
) -> str:
    parts = [
        f"{slugify(parameter.label, fallback='parameter')}-{slugify(value)}"
        for parameter, value in zip(parameters, values, strict=True)
    ]
    full_name = "--".join(parts)
    if len(full_name) <= max_length:
        return full_name
    digest = sha256_text(full_name)[:10]
    return f"{full_name[: max_length - len(digest) - 2].rstrip('-')}--{digest}"


def unique_choice_name(name: str, values: tuple[Any, ...], existing: set[str]) -> str:
    if name not in existing:
        return name
    serialized = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    digest = sha256_text(serialized)[:8]
    candidate = f"{name}--{digest}"
    suffix = 2
    while candidate in existing:
        candidate = f"{name}--{digest}-{suffix}"
        suffix += 1
    return candidate


def screen_name(experiment_id: str, run_name: str, *, max_length: int = 75) -> str:
    full_name = f"rr-{slugify(experiment_id)}--{slugify(run_name)}"
    if len(full_name) <= max_length:
        return full_name
    digest = sha256_text(full_name)[:10]
    return f"{full_name[: max_length - len(digest) - 2].rstrip('-')}--{digest}"


# Compatibility aliases for callers that still use the former sweep terminology.
SWEEP_SCHEMA_VERSION = EXPERIMENT_SCHEMA_VERSION
SweepParameter = ExperimentParameter
SweepRun = ExperimentRun
SweepPlan = ExperimentPlan
