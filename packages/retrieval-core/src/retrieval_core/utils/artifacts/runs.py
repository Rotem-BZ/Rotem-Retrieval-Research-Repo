"""Immutable run artifact manifests and exact upstream run references."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from retrieval_core.utils.hashing import sha256_text
from retrieval_core.utils.io import config_to_yaml, project_path, read_json

MANIFEST_SCHEMA_VERSION = 1
RUN_ID_FORBIDDEN_CHARS = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}


def artifact_for_run(
    cfg: DictConfig,
    *,
    stage_name: str,
    run_id: str,
    artifact_name: str,
) -> Path:
    """Resolve an artifact from an exact upstream run manifest."""

    normalized = str(run_id).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or any(character in normalized for character in RUN_ID_FORBIDDEN_CHARS)
    ):
        raise ValueError(f"Run id must be one directory name, got {run_id!r}.")
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
