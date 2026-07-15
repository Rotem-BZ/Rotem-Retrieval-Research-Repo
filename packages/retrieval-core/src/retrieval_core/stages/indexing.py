"""Indexing stage runner."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_core.pipelines import load_async_pipeline
from retrieval_core.stages.base import StageContext

INDEXING_OUTPUT_COMPONENT = "output"


async def run_indexing(cfg: DictConfig) -> dict:
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
        context.write_manifest(artifacts={"index": index_path})
    return result
