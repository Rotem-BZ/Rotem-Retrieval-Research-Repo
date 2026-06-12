"""Shared stage scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from retrieval_research.io import config_to_yaml, ensure_dir, project_path, write_json, write_text


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
