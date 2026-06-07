"""Indexing stage runner."""

from __future__ import annotations

from omegaconf import DictConfig

from retrieval_research.pipelines import include_outputs, load_async_pipeline, to_container
from retrieval_research.stages.base import StageContext


async def run_indexing(cfg: DictConfig) -> dict:
    pipeline = load_async_pipeline(cfg.pipeline)
    context = StageContext.from_config(cfg)

    result = await pipeline.run_async(
        data=to_container(cfg.pipeline_run.inputs) or {},
        include_outputs_from=include_outputs(cfg.pipeline_run.include_outputs_from),
        concurrency_limit=int(cfg.pipeline_run.concurrency_limit),
    )

    context.write_resolved_config()
    context.write_result(result)
    return result
