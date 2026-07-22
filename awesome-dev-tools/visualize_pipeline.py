"""Render a Haystack pipeline from an immutable stage run configuration."""

from __future__ import annotations

import argparse
from copy import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from retrieval_core.utils.io import read_yaml_mapping
from retrieval_core.utils.pipelines import load_async_pipeline

DEFAULT_SERVER_URL = "https://mermaid.ink"
IMAGE_FORMATS = ("svg", "png", "jpeg", "webp", "pdf")
MERMAID_THEMES = ("default", "neutral", "dark", "forest")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render the Haystack pipeline stored in a resolved stage config."
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
    parser.add_argument("--format", choices=IMAGE_FORMATS, default="svg")
    parser.add_argument("--theme", choices=MERMAID_THEMES, default="neutral")
    parser.add_argument(
        "--background",
        default="!white",
        help="Mermaid background color (default: !white)",
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args(argv)

    try:
        output_path = visualize_pipeline(
            args.resolved_config,
            output_path=args.output,
            image_format=args.format,
            theme=args.theme,
            background=args.background,
            server_url=args.server_url,
            timeout=args.timeout,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(f"Pipeline visualization: {output_path}")


def visualize_pipeline(
    resolved_config_path: Path,
    *,
    output_path: Path | None = None,
    image_format: str = "svg",
    theme: str = "neutral",
    background: str = "!white",
    server_url: str = DEFAULT_SERVER_URL,
    timeout: int = 30,
) -> Path:
    """Load and draw the pipeline embedded in one resolved stage config."""

    config_path = resolved_config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Resolved config does not exist: {config_path}")
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {image_format}")
    if theme not in MERMAID_THEMES:
        raise ValueError(f"Unsupported Mermaid theme: {theme}")
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

    pipeline = pipeline_for_drawing(load_async_pipeline(pipeline_config))
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
    return destination


def pipeline_for_drawing(pipeline: Any) -> Any:
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


def mermaid_params(
    *,
    image_format: str,
    theme: str,
    background: str,
) -> dict[str, Any]:
    """Translate friendly image formats to Mermaid server parameters."""

    params: dict[str, Any] = {
        "format": image_format if image_format in {"svg", "pdf"} else "img",
        "theme": theme,
        "bgColor": background,
    }
    if image_format not in {"svg", "pdf"}:
        params["type"] = image_format
    return params


if __name__ == "__main__":
    main()
