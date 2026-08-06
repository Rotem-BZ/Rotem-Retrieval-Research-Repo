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
from retrieval_core.utils.pipeline_visualization import (
    IMAGE_FORMATS,
    THEMES,
    render_pipeline_visualization,
)

logger = logging.getLogger(__name__)

StageResult = dict[str, Any] | list[dict[str, Any]]
PIPELINE_VISUALIZATION_ARTIFACT = "pipeline_visualization"


class Stage(ABC):
    """Base class for one configured retrieval stage run."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_dir: Path
        self.run_artifacts: dict[str, Path] = {}

    def prepare(self) -> None:
        """Prepare configuration and reserve the immutable run directory."""

        prepare_stage_run_config(self.cfg)
        self.prepare_config()
        visualization_settings = self._pipeline_visualization_settings()
        self.output_dir = project_path(self.cfg.stage.output_dir)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                "Run directory already exists; refusing to overwrite immutable run: "
                f"{self.output_dir}"
            ) from exc
        self.write_resolved_config()
        if visualization_settings is not None:
            self._write_pipeline_visualization(**visualization_settings)

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
        published_artifacts = {**artifacts, **self.run_artifacts}
        path = self.output_dir / "manifest.json"
        written = write_json(
            path,
            run_manifest(self.cfg, artifacts=published_artifacts, inputs=inputs),
        )
        logger.debug("Wrote run manifest: path=%s", written)
        return written

    def _pipeline_visualization_settings(self) -> dict[str, str | None] | None:
        """Validate and return enabled offline-visualization settings."""

        pipeline = self.cfg.get("pipeline")
        stage = self.cfg.get("stage")
        settings = stage.get("visualization") if stage is not None else None
        if pipeline is None or settings is None:
            return None
        if not hasattr(settings, "get"):
            raise TypeError("stage.visualization must be a mapping.")

        enabled = settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("stage.visualization.enabled must be a boolean.")
        if not enabled:
            return None

        image_format = str(settings.get("format", "svg"))
        if image_format not in IMAGE_FORMATS:
            raise ValueError(
                f"stage.visualization.format must be one of {IMAGE_FORMATS}, got {image_format!r}."
            )
        theme = str(settings.get("theme", "neutral"))
        if theme not in THEMES:
            raise ValueError(f"stage.visualization.theme must be one of {THEMES}, got {theme!r}.")
        background = settings.get("background")
        if background is not None and not isinstance(background, str):
            raise TypeError("stage.visualization.background must be a string or null.")
        return {
            "image_format": image_format,
            "theme": theme,
            "background": background,
        }

    def _write_pipeline_visualization(
        self,
        *,
        image_format: str,
        theme: str,
        background: str | None,
    ) -> None:
        """Render a best-effort pipeline image inside the reserved run directory."""

        destination = self.output_dir / f"pipeline.{image_format}"
        title = f"{self.cfg.stage.name} pipeline · {self.cfg.stage.run_id}"
        try:
            written = render_pipeline_visualization(
                self.cfg.pipeline,
                destination,
                image_format=image_format,
                theme=theme,
                background=background,
                title=title,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            logger.warning(
                "Pipeline visualization failed; continuing without it: path=%s",
                destination,
                exc_info=True,
            )
            return

        self.run_artifacts[PIPELINE_VISUALIZATION_ARTIFACT] = written
        logger.info("Pipeline visualization written: path=%s", written)


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
