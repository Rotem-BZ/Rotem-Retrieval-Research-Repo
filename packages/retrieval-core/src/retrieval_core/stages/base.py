"""Shared stage scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

from retrieval_core.utils.artifacts import run_manifest
from retrieval_core.utils.io import config_to_yaml, project_path, write_json, write_text


@dataclass(frozen=True)
class StageContext:
    """Resolved filesystem context for one stage run."""

    cfg: DictConfig
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "StageContext":
        output_dir = project_path(cfg.stage.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Run directory already exists; refusing to overwrite immutable run: {output_dir}"
            ) from exc
        return cls(cfg=cfg, output_dir=output_dir)

    def write_resolved_config(self) -> Path:
        return write_text(self.output_dir / "resolved_config.yaml", config_to_yaml(self.cfg))

    def write_result(self, payload: Any) -> Path:
        return write_json(self.output_dir / "result.json", payload)

    def write_manifest(
        self,
        *,
        artifacts: dict[str, str | Path],
        inputs: dict[str, Any] | None = None,
    ) -> Path:
        path = self.output_dir / "manifest.json"
        return write_json(path, run_manifest(self.cfg, artifacts=artifacts, inputs=inputs))


def prepare_stage_run_config(cfg: DictConfig) -> None:
    """Freeze the run id and update its derived output path."""

    if "stage" not in cfg or "run_id" not in cfg.stage:
        return
    if cfg.stage.get("preserve_run_config", False):
        return

    stage_name = str(cfg.stage.name)
    run_id = str(cfg.stage.run_id)

    with open_dict(cfg):
        cfg.stage.run_id = run_id
        if "paths" in cfg and "runs_dir" in cfg.paths:
            cfg.stage.output_dir = f"{cfg.paths.runs_dir}/{stage_name}/{run_id}"
