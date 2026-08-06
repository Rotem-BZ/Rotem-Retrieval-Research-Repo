from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import yaml

import visualize_pipeline as visualizer


class FakePipeline:
    def __init__(self) -> None:
        self.draw_calls: list[dict[str, Any]] = []
        self.graph = nx.MultiDiGraph()

    def draw(self, **kwargs: Any) -> None:
        self.draw_calls.append(kwargs)


def test_visualizes_resolved_pipeline_offline_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_resolved_config(tmp_path, stage="inference", run_id="baseline")
    render_calls: list[dict[str, Any]] = []

    def render(graph: nx.MultiDiGraph, destination: Path, **kwargs: Any) -> None:
        render_calls.append({"graph": graph, "destination": destination, **kwargs})

    monkeypatch.setattr(visualizer, "render_networkx", render)

    visualizer.main([str(config_path)])

    output_path = (
        tmp_path / "artifacts" / "runs" / "inference" / "baseline" / "pipeline.svg"
    ).resolve()
    assert len(render_calls) == 1
    assert render_calls[0]["destination"] == output_path
    assert render_calls[0]["image_format"] == "svg"
    assert render_calls[0]["theme"] == "neutral"
    assert render_calls[0]["title"] == "inference pipeline · baseline"
    assert set(render_calls[0]["graph"].nodes) == {"input", "retriever", "output"}
    assert output_path.parent.is_dir()
    assert str(output_path) in capsys.readouterr().out


def test_builds_multigraph_directly_from_pipeline_yaml() -> None:
    graph = visualizer.graph_from_pipeline_config(_pipeline_config())

    assert graph.nodes["retriever"] == {
        "component_type": "example.DenseRetriever",
        "label": "retriever\nDense Retriever",
    }
    assert graph.number_of_edges("input", "retriever") == 2
    labels = {data["label"] for _, _, data in graph.edges("input", data=True)}
    assert labels == {
        "query",
        "candidate_document_ids",
    }


def test_collapses_parallel_edges_without_losing_socket_labels() -> None:
    graph = visualizer.graph_from_pipeline_config(_pipeline_config())

    drawing_graph, labels = visualizer.collapsed_drawing_graph(graph)

    assert drawing_graph.number_of_edges("input", "retriever") == 1
    assert labels[("input", "retriever")].splitlines() == [
        "query",
        "candidate_document_ids",
    ]


def test_renders_svg_with_networkx_and_matplotlib(tmp_path: Path) -> None:
    graph = visualizer.graph_from_pipeline_config(_pipeline_config())
    destination = tmp_path / "pipeline.svg"

    visualizer.render_networkx(
        graph,
        destination,
        image_format="svg",
        theme="neutral",
        background=None,
        title="inference pipeline · baseline",
    )

    svg = destination.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "Dense Retriever" in svg
    assert "candidate_document_ids" in svg


def test_supports_explicit_mermaid_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_resolved_config(tmp_path, stage="indexing", run_id="dense")
    output_path = tmp_path / "diagrams" / "dense.webp"
    render_calls: list[dict[str, Any]] = []

    def render(config: dict[str, Any], destination: Path, **kwargs: Any) -> None:
        render_calls.append({"config": config, "destination": destination, **kwargs})

    monkeypatch.setattr(visualizer, "render_mermaid", render)

    result = visualizer.visualize_pipeline(
        config_path,
        output_path=output_path,
        renderer="mermaid",
        image_format="webp",
        theme="dark",
        background="000000",
        server_url="http://localhost:3000",
        timeout=5,
    )

    assert result == output_path.resolve()
    assert render_calls == [
        {
            "config": _pipeline_config(),
            "destination": output_path.resolve(),
            "image_format": "webp",
            "theme": "dark",
            "background": "000000",
            "server_url": "http://localhost:3000",
            "timeout": 5,
        }
    ]


def test_relabels_stage_boundary_components_only_in_mermaid_copy() -> None:
    pipeline = FakePipeline()
    pipeline.graph.add_edges_from(
        [
            ("input", "retriever"),
            ("retriever", "output"),
            ("stage_input", "retriever"),
        ]
    )

    drawing_pipeline = visualizer.pipeline_for_mermaid(pipeline)

    assert set(pipeline.graph.nodes) == {"input", "retriever", "output", "stage_input"}
    assert set(drawing_pipeline.graph.nodes) == {
        "stage_input_2",
        "retriever",
        "stage_output",
        "stage_input",
    }
    assert ("stage_input_2", "retriever") in drawing_pipeline.graph.edges
    assert ("retriever", "stage_output") in drawing_pipeline.graph.edges


def test_uses_deterministic_fallback_layout_for_cycle() -> None:
    graph = nx.DiGraph([("first", "second"), ("second", "first")])

    first_positions, _, _ = visualizer.pipeline_layout(graph)
    second_positions, _, _ = visualizer.pipeline_layout(graph)

    assert first_positions.keys() == second_positions.keys()
    for node in first_positions:
        assert first_positions[node].tolist() == second_positions[node].tolist()


def test_requires_pipeline_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text("stage: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no pipeline mapping"):
        visualizer.visualize_pipeline(config_path, output_path=tmp_path / "pipeline.svg")


def test_rejects_unknown_connection_component() -> None:
    pipeline_config = _pipeline_config()
    pipeline_config["connections"].append(
        {"sender": "missing.value", "receiver": "output.documents"}
    )

    with pytest.raises(ValueError, match="unknown component 'missing'"):
        visualizer.graph_from_pipeline_config(pipeline_config)


def test_requires_output_when_path_is_not_a_stage_run(tmp_path: Path) -> None:
    config_path = tmp_path / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline": _pipeline_config(),
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
                "pipeline": _pipeline_config(),
                "stage": {"name": stage, "run_id": run_id},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _pipeline_config() -> dict[str, Any]:
    return {
        "components": {
            "input": {"type": "example.InferenceInput"},
            "retriever": {"type": "example.DenseRetriever"},
            "output": {"type": "example.InferenceOutput"},
        },
        "connections": [
            {"sender": "input.query", "receiver": "retriever.query"},
            {
                "sender": "input.candidate_document_ids",
                "receiver": "retriever.candidate_document_ids",
            },
            {"sender": "retriever.documents", "receiver": "output.documents"},
        ],
        "metadata": {"description": "Dense retrieval pipeline."},
    }
