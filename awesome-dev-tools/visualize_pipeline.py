"""Render a pipeline graph from an immutable stage run configuration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Any

import networkx as nx

from retrieval_core.utils.io import read_yaml_mapping
from retrieval_core.utils.pipeline_visualization import (
    IMAGE_FORMATS,
    THEMES,
    THEME_COLORS,
    collapsed_drawing_graph,
    graph_from_pipeline_config,
    normalize_background,
    pipeline_layout,
    render_networkx,
    split_endpoint,
    wrap_identifier,
)

DEFAULT_RENDERER = "networkx"
DEFAULT_SERVER_URL = "https://mermaid.ink"
MERMAID_THEMES = THEMES
RENDERERS = ("networkx", "mermaid")

__all__ = [
    "IMAGE_FORMATS",
    "MERMAID_THEMES",
    "RENDERERS",
    "THEME_COLORS",
    "collapsed_drawing_graph",
    "default_output_path",
    "graph_from_pipeline_config",
    "main",
    "mermaid_params",
    "normalize_background",
    "pipeline_for_mermaid",
    "pipeline_layout",
    "pipeline_title",
    "render_mermaid",
    "render_networkx",
    "split_endpoint",
    "visualize_pipeline",
    "wrap_identifier",
]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the pipeline stored in a resolved stage config."
    )
    parser.add_argument("resolved_config", type=Path, help="path to resolved_config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help="output image path; defaults to <run-folder>/pipeline.<format>",
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
    """Place the derived image in the immutable stage run that owns its config."""

    run_dir = resolved_config_path.parent
    stage_dir = run_dir.parent
    runs_dir = stage_dir.parent
    if runs_dir.name != "runs":
        raise ValueError(
            "Cannot infer the stage run from the resolved-config path; pass --output explicitly."
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

    return (run_dir / f"pipeline.{image_format}").resolve()


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
