"""Render a pipeline graph from an immutable stage run configuration."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from pathlib import Path
import re
from typing import Any

import networkx as nx

from retrieval_core.utils.io import read_yaml_mapping

DEFAULT_RENDERER = "networkx"
DEFAULT_SERVER_URL = "https://mermaid.ink"
IMAGE_FORMATS = ("svg", "png", "jpeg", "webp", "pdf")
MERMAID_THEMES = ("default", "neutral", "dark", "forest")
RENDERERS = ("networkx", "mermaid")

THEME_COLORS = {
    "default": {
        "background": "white",
        "component": "#dbeafe",
        "input": "#fef3c7",
        "output": "#dcfce7",
        "border": "#334155",
        "edge": "#475569",
        "text": "#0f172a",
        "edge_label": "#334155",
    },
    "neutral": {
        "background": "white",
        "component": "#e5e7eb",
        "input": "#dbeafe",
        "output": "#dcfce7",
        "border": "#374151",
        "edge": "#6b7280",
        "text": "#111827",
        "edge_label": "#374151",
    },
    "dark": {
        "background": "#111827",
        "component": "#374151",
        "input": "#1e3a8a",
        "output": "#14532d",
        "border": "#d1d5db",
        "edge": "#9ca3af",
        "text": "#f9fafb",
        "edge_label": "#e5e7eb",
    },
    "forest": {
        "background": "#f0fdf4",
        "component": "#dcfce7",
        "input": "#dbeafe",
        "output": "#bbf7d0",
        "border": "#166534",
        "edge": "#15803d",
        "text": "#14532d",
        "edge_label": "#166534",
    },
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the pipeline stored in a resolved stage config."
    )
    parser.add_argument("resolved_config", type=Path, help="path to resolved_config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output image path; defaults to "
            "artifacts/visualizations/pipelines/<stage>/<run-id>.<format>"
        ),
    )
    parser.add_argument("--renderer", choices=RENDERERS, default=DEFAULT_RENDERER)
    parser.add_argument("--format", choices=IMAGE_FORMATS, default="svg")
    parser.add_argument("--theme", choices=MERMAID_THEMES, default="neutral")
    parser.add_argument(
        "--background",
        help="background color; defaults to the selected theme's background",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help="Mermaid renderer URL; ignored by the NetworkX renderer",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Mermaid request timeout; ignored by the NetworkX renderer",
    )
    args = parser.parse_args(argv)

    try:
        output_path = visualize_pipeline(
            args.resolved_config,
            output_path=args.output,
            renderer=args.renderer,
            image_format=args.format,
            theme=args.theme,
            background=args.background,
            server_url=args.server_url,
            timeout=args.timeout,
        )
    except (FileNotFoundError, ImportError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Pipeline visualization: {output_path}")


def visualize_pipeline(
    resolved_config_path: Path,
    *,
    output_path: Path | None = None,
    renderer: str = DEFAULT_RENDERER,
    image_format: str = "svg",
    theme: str = "neutral",
    background: str | None = None,
    server_url: str = DEFAULT_SERVER_URL,
    timeout: int = 30,
) -> Path:
    """Load and draw the pipeline embedded in one resolved stage config."""

    config_path = resolved_config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved config does not exist: {config_path}")
    if renderer not in RENDERERS:
        raise ValueError(f"Unsupported renderer: {renderer}")
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format}")
    if theme not in MERMAID_THEMES:
        raise ValueError(f"Unsupported theme: {theme}")
    if timeout < 1:
        raise ValueError("Timeout must be at least one second.")

    resolved_config = read_yaml_mapping(config_path)
    pipeline_config = resolved_config.get("pipeline")
    if not isinstance(pipeline_config, dict):
        raise ValueError(f"Resolved config has no pipeline mapping: {config_path}")

    destination = (
        output_path.expanduser().resolve()
        if output_path is not None
        else default_output_path(config_path, resolved_config, image_format=image_format)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    if renderer == "networkx":
        graph = graph_from_pipeline_config(pipeline_config)
        render_networkx(
            graph,
            destination,
            image_format=image_format,
            theme=theme,
            background=background,
            title=pipeline_title(resolved_config),
        )
    else:
        render_mermaid(
            pipeline_config,
            destination,
            image_format=image_format,
            theme=theme,
            background=background,
            server_url=server_url,
            timeout=timeout,
        )
    return destination


def graph_from_pipeline_config(pipeline_config: dict[str, Any]) -> nx.MultiDiGraph:
    """Build a graph without importing or initializing configured components."""

    components = pipeline_config.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("Pipeline config must contain a non-empty components mapping.")
    connections = pipeline_config.get("connections", [])
    if not isinstance(connections, list):
        raise ValueError("Pipeline connections must be a list.")

    graph = nx.MultiDiGraph()
    for raw_name, component_config in components.items():
        name = str(raw_name)
        if not isinstance(component_config, dict):
            raise ValueError(f"Component {name!r} must be a mapping.")
        component_type = str(component_config.get("type") or "unknown")
        graph.add_node(
            name,
            component_type=component_type,
            label=f"{name}\n{wrap_identifier(component_type.rsplit('.', 1)[-1])}",
        )

    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            raise ValueError(f"Pipeline connection {index} must be a mapping.")
        sender_component, sender_socket = split_endpoint(connection.get("sender"), index=index)
        receiver_component, receiver_socket = split_endpoint(
            connection.get("receiver"), index=index
        )
        for component_name in (sender_component, receiver_component):
            if component_name not in graph:
                raise ValueError(
                    f"Pipeline connection {index} references unknown component "
                    f"{component_name!r}."
                )
        graph.add_edge(
            sender_component,
            receiver_component,
            sender_socket=sender_socket,
            receiver_socket=receiver_socket,
            label=(
                sender_socket
                if sender_socket == receiver_socket
                else f"{sender_socket}\n→ {receiver_socket}"
            ),
        )

    metadata = pipeline_config.get("metadata")
    if isinstance(metadata, dict) and metadata.get("description"):
        graph.graph["description"] = str(metadata["description"])
    return graph


def split_endpoint(value: Any, *, index: int) -> tuple[str, str]:
    """Split a Haystack connection endpoint into component and socket names."""

    if not isinstance(value, str) or "." not in value:
        raise ValueError(
            f"Pipeline connection {index} endpoints must use '<component>.<socket>'."
        )
    component_name, socket_name = value.split(".", 1)
    if not component_name or not socket_name:
        raise ValueError(
            f"Pipeline connection {index} endpoints must use '<component>.<socket>'."
        )
    return component_name, socket_name


def render_networkx(
    graph: nx.MultiDiGraph,
    destination: Path,
    *,
    image_format: str,
    theme: str,
    background: str | None,
    title: str,
) -> None:
    """Render a pipeline locally with NetworkX and headless Matplotlib."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "The NetworkX renderer requires Matplotlib; sync the project's dev extra."
        ) from error

    colors = dict(THEME_COLORS[theme])
    colors["background"] = normalize_background(background) or colors["background"]
    layout_graph, edge_labels = collapsed_drawing_graph(graph)
    positions, layer_count, max_layer_size = pipeline_layout(layout_graph)
    max_layer_span = max(
        (
            round(abs(positions[receiver][0] - positions[sender][0]) / 4.0)
            for sender, receiver in layout_graph.edges
        ),
        default=1,
    )
    width = max(9.0, 3.0 * layer_count)
    height = max(
        3.2 if max_layer_span == 1 else 4.5,
        1.7 * max_layer_size + 1.2,
    )

    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure, axis = plt.subplots(figsize=(width, height), facecolor=colors["background"])
        axis.set_facecolor(colors["background"])
        for edge_index, (sender, receiver) in enumerate(layout_graph.edges):
            layer_span = max(1, round(abs(positions[receiver][0] - positions[sender][0]) / 4.0))
            curve = 0.0 if layer_span == 1 else min(0.08 + 0.025 * layer_span, 0.18)
            if edge_index % 2:
                curve = -curve
            connection_style = f"arc3,rad={curve}"
            nx.draw_networkx_edges(
                layout_graph,
                positions,
                edgelist=[(sender, receiver)],
                ax=axis,
                arrows=True,
                arrowsize=18,
                arrowstyle="-|>",
                connectionstyle=connection_style,
                edge_color=colors["edge"],
                node_size=0,
                min_source_margin=58,
                min_target_margin=58,
                width=1.4,
            )
            nx.draw_networkx_edge_labels(
                layout_graph,
                positions,
                edge_labels={(sender, receiver): edge_labels[(sender, receiver)]},
                ax=axis,
                font_size=7,
                font_color=colors["edge_label"],
                rotate=False,
                connectionstyle=connection_style,
                bbox={
                    "alpha": 0.94,
                    "edgecolor": "none",
                    "facecolor": colors["background"],
                    "pad": 1.5,
                },
            )
        for node, (x_position, y_position) in positions.items():
            node_color = (
                colors["input"]
                if node == "input"
                else colors["output"]
                if node == "output"
                else colors["component"]
            )
            axis.text(
                x_position,
                y_position,
                str(layout_graph.nodes[node]["label"]),
                horizontalalignment="center",
                verticalalignment="center",
                color=colors["text"],
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.65",
                    "edgecolor": colors["border"],
                    "facecolor": node_color,
                    "linewidth": 1.4,
                },
                zorder=3,
            )
        axis.set_title(title, color=colors["text"], fontsize=12, pad=18, loc="left")
        x_values = [position[0] for position in positions.values()]
        y_values = [position[1] for position in positions.values()]
        axis.set_xlim(min(x_values) - 1.7, max(x_values) + 1.7)
        vertical_margin = 1.2 if max_layer_span == 1 else 1.8
        axis.set_ylim(min(y_values) - vertical_margin, max(y_values) + vertical_margin)
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(
            destination,
            format=image_format,
            dpi=180,
            bbox_inches="tight",
            facecolor=colors["background"],
        )
        plt.close(figure)


def collapsed_drawing_graph(
    graph: nx.MultiDiGraph,
) -> tuple[nx.DiGraph, dict[tuple[str, str], str]]:
    """Collapse parallel component edges while retaining every socket mapping."""

    drawing_graph = nx.DiGraph()
    drawing_graph.add_nodes_from(graph.nodes(data=True))
    labels_by_edge: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for sender, receiver, edge_data in graph.edges(data=True):
        drawing_graph.add_edge(sender, receiver)
        labels_by_edge[(sender, receiver)].append(str(edge_data["label"]))
    edge_labels = {
        edge: "\n".join(labels)
        for edge, labels in labels_by_edge.items()
    }
    return drawing_graph, edge_labels


def pipeline_layout(graph: nx.DiGraph) -> tuple[dict[str, Any], int, int]:
    """Lay out DAGs in execution layers and fall back deterministically for cycles."""

    if nx.is_directed_acyclic_graph(graph):
        generations = [sorted(generation) for generation in nx.topological_generations(graph)]
        positions = {
            node: (4.0 * layer, 2.2 * (index - (len(nodes) - 1) / 2))
            for layer, nodes in enumerate(generations)
            for index, node in enumerate(nodes)
        }
        return positions, max(1, len(generations)), max(map(len, generations), default=1)
    positions = nx.spring_layout(graph, seed=42)
    return positions, max(1, len(graph)), max(1, len(graph) // 2)


def wrap_identifier(identifier: str, *, line_width: int = 22) -> str:
    """Wrap CamelCase and snake_case identifiers without truncating them."""

    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier).replace("_", " ").split()
    if not words:
        return identifier
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= line_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def render_mermaid(
    pipeline_config: dict[str, Any],
    destination: Path,
    *,
    image_format: str,
    theme: str,
    background: str | None,
    server_url: str,
    timeout: int,
) -> None:
    """Render through a Mermaid-compatible HTTP service."""

    from retrieval_core.utils.pipelines import load_async_pipeline

    pipeline = pipeline_for_mermaid(load_async_pipeline(pipeline_config))
    pipeline.draw(
        path=destination,
        server_url=server_url,
        params=mermaid_params(
            image_format=image_format,
            theme=theme,
            background=background,
        ),
        timeout=timeout,
    )


def pipeline_for_mermaid(pipeline: Any) -> Any:
    """Copy a pipeline and disambiguate names reserved by Haystack's renderer."""

    reserved_names = {"input", "output"}
    existing_names = set(pipeline.graph.nodes)
    replacements: dict[str, str] = {}
    for name in reserved_names & existing_names:
        candidate = f"stage_{name}"
        suffix = 2
        while candidate in existing_names:
            candidate = f"stage_{name}_{suffix}"
            suffix += 1
        replacements[name] = candidate
        existing_names.add(candidate)

    drawing_pipeline = copy(pipeline)
    drawing_pipeline.graph = nx.relabel_nodes(
        pipeline.graph,
        replacements,
        copy=True,
    )
    return drawing_pipeline


def default_output_path(
    resolved_config_path: Path,
    resolved_config: dict[str, Any],
    *,
    image_format: str,
) -> Path:
    """Place derived images beside, but not inside, immutable stage run trees."""

    run_dir = resolved_config_path.parent
    stage_dir = run_dir.parent
    runs_dir = stage_dir.parent
    artifacts_dir = runs_dir.parent
    if runs_dir.name != "runs" or artifacts_dir.name != "artifacts":
        raise ValueError(
            "Cannot infer the visualization directory from the resolved-config path; "
            "pass --output explicitly."
        )

    stage = resolved_config.get("stage")
    if not isinstance(stage, dict):
        raise ValueError("Resolved config has no stage mapping; pass --output explicitly.")
    stage_name = str(stage.get("name") or "")
    run_id = str(stage.get("run_id") or "")
    if stage_name != stage_dir.name or run_id != run_dir.name:
        raise ValueError(
            "Resolved stage name or run id does not match its run directory; "
            "pass --output explicitly."
        )

    return (
        artifacts_dir
        / "visualizations"
        / "pipelines"
        / stage_name
        / f"{run_id}.{image_format}"
    ).resolve()


def pipeline_title(resolved_config: dict[str, Any]) -> str:
    """Create a concise title from immutable run identity when available."""

    stage = resolved_config.get("stage")
    if not isinstance(stage, dict):
        return "Haystack pipeline"
    stage_name = str(stage.get("name") or "").strip()
    run_id = str(stage.get("run_id") or "").strip()
    if stage_name and run_id:
        return f"{stage_name} pipeline · {run_id}"
    return f"{stage_name} pipeline" if stage_name else "Haystack pipeline"


def normalize_background(background: str | None) -> str | None:
    """Accept Mermaid's named-color marker in the local renderer too."""

    if background is None:
        return None
    return background[1:] if background.startswith("!") else background


def mermaid_params(
    *,
    image_format: str,
    theme: str,
    background: str | None,
) -> dict[str, Any]:
    """Translate friendly image formats to Mermaid server parameters."""

    mermaid_background = background or "!white"
    if not mermaid_background.startswith(("!", "#")):
        mermaid_background = f"!{mermaid_background}"
    params: dict[str, Any] = {
        "format": image_format if image_format in {"svg", "pdf"} else "img",
        "theme": theme,
        "bgColor": mermaid_background,
    }
    if image_format not in {"svg", "pdf"}:
        params["type"] = image_format
    return params


if __name__ == "__main__":
    main()
