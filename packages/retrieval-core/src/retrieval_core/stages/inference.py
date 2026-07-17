"""Inference stage runner."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack import AsyncPipeline, Document
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from retrieval_core.input_mapping import (
    InferenceMapping,
    resolve_inference_mapping,
    validate_input_mapping_config,
)
from retrieval_core.stages.base import StageContext, is_dry_run
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

    if is_dry_run(cfg):
        predictions_path = cfg.stage.predictions_path
    else:
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
    query_id = str(query["id"])
    inputs = _build_query_inputs(
        query["text"],
        candidate_document_ids=inference_mapping.candidate_ids(query_id),
        candidate_documents=inference_mapping.candidate_documents(query_id),
    )
    result = await pipeline.run_async(
        data=inputs,
        include_outputs_from={INFERENCE_OUTPUT_COMPONENT},
        concurrency_limit=pipeline_concurrency_limit,
    )
    documents = _extract_documents(result)
    return {
        "query_id": query_id,
        "query": query["text"],
        "documents": [_document_to_dict(document) for document in documents],
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


def validate_inference_inputs(cfg: DictConfig) -> None:
    """Validate file-backed pipeline inputs without executing the pipeline."""

    for component_name, component_cfg in cfg.pipeline.get("components", {}).items():
        init_parameters = component_cfg.get("init_parameters", {})
        if "index_path" not in init_parameters:
            continue
        index_path = init_parameters.get("index_path")
        if not index_path:
            raise ValueError(
                f"Pipeline component {component_name!r} requires an index. Set either "
                "stage.indexing_run_id to an exact indexing run id or stage.index_path."
            )
        resolved = project_path(index_path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Index for component {component_name!r} does not exist: {resolved}"
            )


def _build_query_inputs(
    query_text: str,
    *,
    candidate_document_ids: list[str] | None = None,
    candidate_documents: list[Document] | None = None,
) -> dict[str, Any]:
    return {
        INFERENCE_INPUT_COMPONENT: {
            "query": query_text,
            "candidate_document_ids": list(candidate_document_ids or []),
            "candidate_documents": list(candidate_documents or []),
        }
    }


def _extract_documents(result: dict[str, Any]) -> list[Document]:
    documents = result[INFERENCE_OUTPUT_COMPONENT][INFERENCE_DOCUMENTS_FIELD]
    return list(documents)


def _document_to_dict(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "content": document.content,
        "meta": dict(document.meta or {}),
        "score": getattr(document, "score", None),
    }
