"""Immutable run artifact manifests and exact upstream run references."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from yaml import YAMLError

from retrieval_core.utils.hashing import sha256_text
from retrieval_core.utils.io import config_to_yaml, project_path, read_json, read_yaml_mapping

MANIFEST_SCHEMA_VERSION = 1
RUN_ID_FORBIDDEN_CHARS = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}


def discover_inference_run_ids(
    runs_dir: str | Path,
    *,
    dataset_name: str,
) -> list[str]:
    """Return completed inference run ids compatible with one dataset."""

    root = project_path(runs_dir) / "inference"
    if not root.is_dir():
        return []

    run_ids: list[str] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        manifest_stage = manifest.get("stage")
        if isinstance(manifest_stage, dict) and manifest_stage.get("name") not in {
            None,
            "inference",
        }:
            continue
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        raw_predictions_path = artifacts.get("predictions")
        if not raw_predictions_path or not project_path(raw_predictions_path).is_file():
            continue
        if _inference_run_dataset(directory, manifest) != str(dataset_name):
            continue
        try:
            run_ids.append(_validate_run_id(directory.name))
        except ValueError:
            continue
    return sorted(run_ids)


def _inference_run_dataset(directory: Path, manifest: dict[str, Any]) -> str | None:
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict) and inputs.get("dataset") is not None:
        return str(inputs["dataset"])

    resolved_config_path = directory / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        return None
    try:
        resolved_config = read_yaml_mapping(resolved_config_path)
    except (OSError, ValueError, YAMLError):
        return None
    dataset = resolved_config.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("name") is None:
        return None
    return str(dataset["name"])


def _validate_run_id(run_id: object) -> str:
    normalized = str(run_id).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or any(character in normalized for character in RUN_ID_FORBIDDEN_CHARS)
    ):
        raise ValueError(f"Run id must be one directory name, got {run_id!r}.")
    return normalized


def artifact_for_run(
    cfg: DictConfig,
    *,
    stage_name: str,
    run_id: str,
    artifact_name: str,
) -> Path:
    """Resolve an artifact from an exact upstream run manifest."""

    normalized = _validate_run_id(run_id)
    directory = project_path(cfg.paths.runs_dir) / stage_name / normalized
    if not directory.is_dir():
        raise FileNotFoundError(f"No {stage_name} run exists with id {run_id!r}: {directory}")

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run {stage_name}/{run_id} has no manifest.json. "
            "Use an explicit artifact path for legacy runs."
        )

    manifest = read_json(manifest_path)
    raw_path = manifest.get("artifacts", {}).get(artifact_name)
    if not raw_path:
        raise KeyError(f"Run {stage_name}/{run_id} does not declare artifact {artifact_name!r}.")

    artifact_path = project_path(raw_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact {artifact_name!r} declared by {stage_name}/{run_id} does not exist: "
            f"{artifact_path}"
        )
    return artifact_path


def run_manifest(
    cfg: DictConfig,
    *,
    artifacts: dict[str, str | Path],
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact provenance manifest for one completed stage run."""

    resolved_config = config_to_yaml(cfg)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": {
            "name": str(cfg.stage.name),
            "run_id": str(cfg.stage.run_id),
        },
        "artifacts": {name: str(project_path(path)) for name, path in artifacts.items()},
        "inputs": dict(inputs or {}),
        "configuration": {
            "sha256": sha256_text(resolved_config),
        },
        "code": {
            "git_commit": _git_commit(),
            "packages": {
                "retrieval-core": _package_version("retrieval-core"),
                "retrieval-components": _package_version("retrieval-components"),
            },
            "python_version": sys.version.split()[0],
        },
    }
    experiment = cfg.get("experiment")
    if experiment:
        parameters = experiment.get("parameters", {})
        if OmegaConf.is_config(parameters):
            parameters = OmegaConf.to_container(parameters, resolve=True)
        manifest["experiment"] = {
            "id": str(experiment.get("id")),
            "name": str(experiment.get("name")),
            "run_name": str(experiment.get("run_name")),
            "parameters": parameters,
        }
    return manifest


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path("."),
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
