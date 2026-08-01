"""Shared lifecycle for retrieval experiment stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

from retrieval_core.utils.artifacts import run_manifest
from retrieval_core.utils.io import config_to_yaml, project_path, write_json, write_text

logger = logging.getLogger(__name__)

StageResult = dict[str, Any] | list[dict[str, Any]]


class Stage(ABC):
    """Base class for one configured retrieval stage run."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_dir: Path

    def prepare(self) -> None:
        """Prepare configuration and reserve the immutable run directory."""

        prepare_stage_run_config(self.cfg)
        self.prepare_config()
        self.output_dir = project_path(self.cfg.stage.output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                "Run directory already exists; refusing to overwrite immutable run: "
                f"{self.output_dir}"
            ) from exc
        self.write_resolved_config()

    @abstractmethod
    def prepare_config(self) -> None:
        """Resolve dependencies and validate configuration before material writes."""

    @abstractmethod
    def run(self) -> StageResult | Awaitable[StageResult]:
        """Execute the configured stage and return its compact result."""

    def write_resolved_config(self) -> Path:
        path = write_text(self.output_dir / "resolved_config.yaml", config_to_yaml(self.cfg))
        logger.debug("Wrote resolved configuration: path=%s", path)
        return path

    def write_result(self, payload: Any) -> Path:
        path = write_json(self.output_dir / "result.json", payload)
        logger.debug("Wrote stage result: path=%s", path)
        return path

    def write_manifest(
        self,
        *,
        artifacts: dict[str, str | Path],
        inputs: dict[str, Any] | None = None,
    ) -> Path:
        path = self.output_dir / "manifest.json"
        written = write_json(
            path,
            run_manifest(self.cfg, artifacts=artifacts, inputs=inputs),
        )
        logger.debug("Wrote run manifest: path=%s", written)
        return written


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
