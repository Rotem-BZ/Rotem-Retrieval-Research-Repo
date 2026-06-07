"""Shared stage scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from retrieval_research.io import config_to_yaml, ensure_dir, write_json, write_text


@dataclass(frozen=True)
class StageContext:
    """Resolved filesystem context for one stage run."""

    cfg: DictConfig
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "StageContext":
        return cls(cfg=cfg, output_dir=ensure_dir(cfg.stage.output_dir))

    def write_resolved_config(self) -> Path:
        return write_text(self.output_dir / "resolved_config.yaml", config_to_yaml(self.cfg))

    def write_result(self, payload: Any) -> Path:
        return write_json(self.output_dir / "result.json", payload)
