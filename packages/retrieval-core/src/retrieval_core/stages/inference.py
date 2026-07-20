"""Inference stage runner."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack import AsyncPipeline
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.input_mapping import (
    InferenceMapping,
    resolve_inference_mapping,
    validate_input_mapping_config,
)
from retrieval_core.stages.base import StageContext
from retrieval_core.utils.artifacts import artifact_for_run
from retrieval_core.utils.io import project_path, write_predictions
from retrieval_core.utils.pipelines import load_async_pipeline

INFERENCE_INPUT_COMPONENT = "input"
INFERENCE_OUTPUT_COMPONENT = "output"
INFERENCE_DOCUMENTS_FIELD = "documents"


async def run_inference(cfg: DictConfig) -> list[dict[str, Any]]:
    prepare_inference_config(cfg)
    pipeline = load_async_pipeline(cfg.pipeline)
    context = StageContext.from_config(cfg)

    inference_mapping = resolve_inference_mapping(cfg)
    pipeline_concurrency_limit = int(cfg.runtime.concurrency_limit)
    query_concurrency_limit = int(cfg.runtime.query_concurrency_limit)
    predictions = await _run_queries(
        pipeline,
        inference_mapping,
        query_concurrency_limit=query_concurrency_limit,
        pipeline_concurrency_limit=pipeline_concurrency_limit,
    )

    predictions_path = write_predictions(cfg.stage.predictions_path, predictions)

    context.write_resolved_config()
    context.write_result(
        {
            "predictions_path": str(predictions_path),
            "query_count": len(predictions),
        },
    )
    inputs: dict[str, Any] = {}
    if cfg.stage.get("indexing_run_id"):
        inputs["indexing_run_id"] = str(cfg.stage.indexing_run_id)
    if cfg.stage.get("index_path"):
        inputs["index_path"] = str(project_path(cfg.stage.index_path))
    input_mapping_path = validate_input_mapping_config(cfg)
    if input_mapping_path is not None:
        inputs["input_mapping_path"] = str(input_mapping_path)
    context.write_manifest(
        artifacts={"predictions": predictions_path},
        inputs=inputs,
    )
    return predictions


async def _run_queries(
    pipeline: AsyncPipeline,
    inference_mapping: InferenceMapping,
    *,
    query_concurrency_limit: int,
    pipeline_concurrency_limit: int,
) -> list[dict[str, Any]]:
    """Run independent queries concurrently while retaining their input order."""

    if query_concurrency_limit < 1:
        raise ValueError("runtime.query_concurrency_limit must be at least 1.")

    queries = inference_mapping.queries
    if not queries:
        return []

    pending_queries = iter(enumerate(queries))
    ordered_predictions: list[dict[str, Any] | None] = [None] * len(queries)

    async def worker(progress: tqdm[Any]) -> None:
        for query_index, query in pending_queries:
            ordered_predictions[query_index] = await _run_query(
                pipeline,
                inference_mapping,
                query,
                pipeline_concurrency_limit=pipeline_concurrency_limit,
            )
            progress.update()

    with tqdm(total=len(queries), desc="queries") as progress:
        workers = [
            asyncio.create_task(worker(progress))
            for _ in range(min(query_concurrency_limit, len(queries)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

    return [prediction for prediction in ordered_predictions if prediction is not None]


async def _run_query(
    pipeline: AsyncPipeline,
    inference_mapping: InferenceMapping,
    query: dict[str, Any],
    *,
    pipeline_concurrency_limit: int,
) -> dict[str, Any]:
    EVALUATION_DATA_SCHEMA.validate_query(query)
    query_id = str(query[EVALUATION_DATA_SCHEMA.query_id])
    query_input = str(query[EVALUATION_DATA_SCHEMA.IN])
    query_content = str(query[EVALUATION_DATA_SCHEMA.query_content])
    candidate_document_ids = list(inference_mapping.candidate_ids(query_input))
    result = await pipeline.run_async(
        data={
            INFERENCE_INPUT_COMPONENT: {
                "query": query_content,
                "candidate_document_ids": candidate_document_ids,
                "candidate_documents": [
                    inference_mapping.documents_by_id[document_id]
                    for document_id in candidate_document_ids
                ],
            }
        },
        include_outputs_from={INFERENCE_OUTPUT_COMPONENT},
        concurrency_limit=pipeline_concurrency_limit,
    )
    documents = list(result[INFERENCE_OUTPUT_COMPONENT][INFERENCE_DOCUMENTS_FIELD])
    return {
        EVALUATION_DATA_SCHEMA.query_id: query_id,
        EVALUATION_DATA_SCHEMA.IN: query_input,
        EVALUATION_DATA_SCHEMA.query_content: query_content,
        "documents": [
            {
                "id": document.id,
                "content": document.content,
                "meta": dict(document.meta or {}),
                "score": getattr(document, "score", None),
            }
            for document in documents
        ],
    }


def prepare_inference_config(cfg: DictConfig) -> None:
    """Resolve an exact indexing run reference into the configured index path."""

    indexing_run_id = cfg.stage.get("indexing_run_id")
    if indexing_run_id:
        resolved = artifact_for_run(
            cfg,
            stage_name="indexing",
            run_id=str(indexing_run_id),
            artifact_name="index",
        )
        configured_path = cfg.stage.get("index_path")
        if configured_path and project_path(configured_path) != resolved:
            raise ValueError(
                "stage.indexing_run_id and stage.index_path resolve to different artifacts."
            )
        with open_dict(cfg):
            cfg.stage.index_path = str(resolved)
