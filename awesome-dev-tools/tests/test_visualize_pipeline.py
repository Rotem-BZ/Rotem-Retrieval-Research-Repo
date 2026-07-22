from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
import networkx as nx

import visualize_pipeline as visualizer


class FakePipeline:
    def __init__(self) -> None:
        self.draw_calls: list[dict[str, Any]] = []
        self.graph = nx.MultiDiGraph()

    def draw(self, **kwargs: Any) -> None:
        self.draw_calls.append(kwargs)


def test_visualizes_resolved_pipeline_outside_immutable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_resolved_config(tmp_path, stage="inference", run_id="baseline")
    pipeline = FakePipeline()
    loaded_configs: list[dict[str, Any]] = []

    def load(config: dict[str, Any]) -> FakePipeline:
        loaded_configs.append(config)
        return pipeline

    monkeypatch.setattr(visualizer, "load_async_pipeline", load)

    visualizer.main([str(config_path)])

    output_path = (
        tmp_path
        / "artifacts"
        / "visualizations"
        / "pipelines"
        / "inference"
        / "baseline.svg"
    ).resolve()
    assert loaded_configs == [{"components": {}, "connections": []}]
    assert pipeline.draw_calls == [
        {
            "path": output_path,
            "server_url": visualizer.DEFAULT_SERVER_URL,
            "params": {
                "format": "svg",
                "theme": "neutral",
                "bgColor": "!white",
            },
            "timeout": 30,
        }
    ]
    assert output_path.parent.is_dir()
    assert str(output_path) in capsys.readouterr().out


def test_supports_explicit_output_and_raster_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resolved_config(tmp_path, stage="indexing", run_id="dense")
    output_path = tmp_path / "diagrams" / "dense.webp"
    pipeline = FakePipeline()
    monkeypatch.setattr(visualizer, "load_async_pipeline", lambda _config: pipeline)

    result = visualizer.visualize_pipeline(
        config_path,
        output_path=output_path,
        image_format="webp",
        theme="dark",
        background="000000",
        server_url="http://localhost:3000",
        timeout=5,
    )

    assert result == output_path.resolve()
    assert pipeline.draw_calls[0] == {
        "path": output_path.resolve(),
        "server_url": "http://localhost:3000",
        "params": {
            "format": "img",
            "type": "webp",
            "theme": "dark",
            "bgColor": "000000",
        },
        "timeout": 5,
    }


def test_requires_pipeline_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text("stage: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no pipeline mapping"):
        visualizer.visualize_pipeline(config_path, output_path=tmp_path / "pipeline.svg")


def test_relabels_stage_boundary_components_only_in_drawing_copy() -> None:
    pipeline = FakePipeline()
    pipeline.graph.add_edges_from(
        [
            ("input", "retriever"),
            ("retriever", "output"),
            ("stage_input", "retriever"),
        ]
    )

    drawing_pipeline = visualizer.pipeline_for_drawing(pipeline)

    assert set(pipeline.graph.nodes) == {"input", "retriever", "output", "stage_input"}
    assert set(drawing_pipeline.graph.nodes) == {
        "stage_input_2",
        "retriever",
        "stage_output",
        "stage_input",
    }
    assert ("stage_input_2", "retriever") in drawing_pipeline.graph.edges
    assert ("retriever", "stage_output") in drawing_pipeline.graph.edges


def test_requires_output_when_path_is_not_a_stage_run(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"components": {}, "connections": []},
                "stage": {"name": "inference", "run_id": "baseline"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pass --output explicitly"):
        visualizer.visualize_pipeline(config_path)


def _write_resolved_config(root: Path, *, stage: str, run_id: str) -> Path:
    config_path = root / "artifacts" / "runs" / stage / run_id / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": {"components": {}, "connections": []},
                "stage": {"name": stage, "run_id": run_id},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path
