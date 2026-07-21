"""Indexing stage runner."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_core.stages.base import StageContext
from retrieval_core.utils.artifacts import index_artifact_path
from retrieval_core.utils.io import project_path
from retrieval_core.utils.pipelines import load_async_pipeline

INDEXING_OUTPUT_COMPONENT = "output"


async def run_indexing(cfg: DictConfig) -> dict:
    index_id = str(cfg.selections.index_id)
    canonical_index_path = index_artifact_path(cfg.paths.indexes_dir, index_id)
    configured_index_path = project_path(_configured_index_output_path(cfg))
    if configured_index_path != canonical_index_path:
        raise ValueError(
            "The indexing pipeline output_path must resolve from "
            "paths.indexes_dir and selections.index_id."
        )
    if canonical_index_path.exists():
        raise FileExistsError(
            f"Index {index_id!r} already exists; choose another selections.index_id: "
            f"{canonical_index_path}"
        )

    pipeline = load_async_pipeline(cfg.pipeline)
    context = StageContext.from_config(cfg)

    result = await pipeline.run_async(
        data={},
        include_outputs_from={INDEXING_OUTPUT_COMPONENT},
        concurrency_limit=int(cfg.runtime.concurrency_limit),
    )

    context.write_resolved_config()
    context.write_result(result)
    output = result.get(INDEXING_OUTPUT_COMPONENT, {})
    index_path = output.get("index_path")
    if index_path:
        if project_path(index_path) != canonical_index_path:
            raise RuntimeError(
                f"Indexing pipeline returned a non-canonical index path: {index_path}"
            )
        context.write_manifest(
            artifacts={"index": index_path},
            inputs={"index_id": index_id},
        )
    return result


def _configured_index_output_path(cfg: DictConfig) -> str:
    """Return the output path for the component connected to output.index_path."""

    senders = [
        str(connection.sender)
        for connection in cfg.pipeline.connections
        if str(connection.receiver) == f"{INDEXING_OUTPUT_COMPONENT}.index_path"
    ]
    if len(senders) != 1:
        raise ValueError(
            "The indexing pipeline must connect exactly one index_path to output.index_path."
        )
    component_name = senders[0].split(".", 1)[0]
    init_parameters = cfg.pipeline.components[component_name].get("init_parameters")
    if init_parameters is None or not init_parameters.get("output_path"):
        raise ValueError(
            "The component connected to output.index_path must declare output_path."
        )
    return str(init_parameters.output_path)
