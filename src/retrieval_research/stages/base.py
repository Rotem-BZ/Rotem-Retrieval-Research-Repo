"""Shared stage scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

from retrieval_research.io import config_to_yaml, ensure_dir, project_path, write_json, write_text

RUN_NAME_FORBIDDEN_CHARS = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}


@dataclass(frozen=True)
class StageContext:
    """Resolved filesystem context for one stage run."""

    cfg: DictConfig
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "StageContext":
        if is_dry_run(cfg):
            return cls(cfg=cfg, output_dir=project_path(cfg.stage.output_dir))
        return cls(cfg=cfg, output_dir=ensure_dir(cfg.stage.output_dir))

    def write_resolved_config(self) -> Path:
        if is_dry_run(self.cfg):
            return self.output_dir / "resolved_config.yaml"
        return write_text(self.output_dir / "resolved_config.yaml", config_to_yaml(self.cfg))

    def write_result(self, payload: Any) -> Path:
        if is_dry_run(self.cfg):
            return self.output_dir / "result.json"
        return write_json(self.output_dir / "result.json", payload)


def is_dry_run(cfg: DictConfig) -> bool:
    return bool(cfg.stage.get("dry_run", False))


def prepare_stage_run_config(cfg: DictConfig) -> None:
    """Freeze the run id and apply optional user-friendly run naming."""

    if "stage" not in cfg or "run_id" not in cfg.stage:
        return

    stage_name = str(cfg.stage.name)
    run_id = named_run_id(cfg.stage.get("run_name"), str(cfg.stage.run_id))

    with open_dict(cfg):
        cfg.stage.run_id = run_id
        if "paths" in cfg and "runs_dir" in cfg.paths:
            cfg.stage.output_dir = f"{cfg.paths.runs_dir}/{stage_name}/{run_id}"
        if "hydra" in cfg and "run" in cfg.hydra and "dir" in cfg.hydra.run and "paths" in cfg:
            cfg.hydra.run.dir = f"{cfg.paths.runs_dir}/hydra/{stage_name}/{run_id}"


def named_run_id(run_name: Any, timestamp_run_id: str) -> str:
    if run_name is None:
        return timestamp_run_id

    normalized = str(run_name).strip()
    if not normalized:
        return timestamp_run_id
    if any(char in normalized for char in RUN_NAME_FORBIDDEN_CHARS):
        forbidden = "".join(sorted(RUN_NAME_FORBIDDEN_CHARS))
        raise ValueError(f"stage.run_name must not contain path characters: {forbidden}")

    return f"{normalized}_{timestamp_run_id}"
