from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.stages.base import PIPELINE_VISUALIZATION_ARTIFACT, Stage
from retrieval_core.utils.io import read_json


class ExampleStage(Stage):
    def prepare_config(self) -> None:
        pass

    def run(self) -> dict:
        return {"ok": True}


def test_stage_renders_pipeline_and_publishes_it_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    render_calls: list[dict] = []

    def render(pipeline_config, destination, **kwargs):
        output_path = Path(destination).resolve()
        output_path.write_text("<svg />", encoding="utf-8")
        render_calls.append(
            {"pipeline_config": pipeline_config, "destination": output_path, **kwargs}
        )
        return output_path

    monkeypatch.setattr("retrieval_core.stages.base.render_pipeline_visualization", render)

    stage = ExampleStage(cfg)
    stage.prepare()
    stage.write_manifest(artifacts={"result": stage.output_dir / "result.json"})

    visualization_path = stage.output_dir / "pipeline.svg"
    assert visualization_path.read_text(encoding="utf-8") == "<svg />"
    assert render_calls[0]["destination"] == visualization_path
    assert render_calls[0]["image_format"] == "svg"
    assert render_calls[0]["theme"] == "neutral"
    assert render_calls[0]["background"] is None
    assert render_calls[0]["title"] == "inference pipeline · automatic"
    manifest = read_json(stage.output_dir / "manifest.json")
    assert manifest["artifacts"][PIPELINE_VISUALIZATION_ARTIFACT] == str(visualization_path)


def test_stage_continues_when_pipeline_rendering_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _config(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("renderer exploded")

    monkeypatch.setattr("retrieval_core.stages.base.render_pipeline_visualization", fail)
    caplog.set_level("WARNING", logger="retrieval_core.stages.base")

    stage = ExampleStage(cfg)
    stage.prepare()
    stage.write_manifest(artifacts={})

    assert not (stage.output_dir / "pipeline.svg").exists()
    assert (
        PIPELINE_VISUALIZATION_ARTIFACT
        not in read_json(stage.output_dir / "manifest.json")["artifacts"]
    )
    assert "Pipeline visualization failed; continuing without it" in caplog.text
    assert "RuntimeError: renderer exploded" in caplog.text


def test_stage_skips_disabled_pipeline_visualization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(tmp_path)
    cfg.stage.visualization.enabled = False

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled renderer was called")

    monkeypatch.setattr("retrieval_core.stages.base.render_pipeline_visualization", unexpected)

    stage = ExampleStage(cfg)
    stage.prepare()

    assert not (stage.output_dir / "pipeline.svg").exists()


def test_invalid_pipeline_visualization_config_fails_before_creating_run(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg.stage.visualization.format = "html"
    expected_output_dir = tmp_path / "artifacts" / "runs" / "inference" / "automatic"

    with pytest.raises(ValueError, match="stage.visualization.format"):
        ExampleStage(cfg).prepare()

    assert not expected_output_dir.exists()


def _config(tmp_path: Path):
    return OmegaConf.create(
        {
            "paths": {"runs_dir": str(tmp_path / "artifacts" / "runs")},
            "pipeline": {
                "components": {
                    "input": {"type": "example.Input"},
                    "output": {"type": "example.Output"},
                },
                "connections": [
                    {"sender": "input.value", "receiver": "output.value"},
                ],
            },
            "stage": {
                "name": "inference",
                "run_id": "automatic",
                "output_dir": "ignored-by-preparation",
                "visualization": {
                    "enabled": True,
                    "format": "svg",
                    "theme": "neutral",
                    "background": None,
                },
            },
        }
    )
