"""Inference stage runner."""

from __future__ import annotations

import asyncio
from typing import Any

from haystack import AsyncPipeline
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.input_mapping import (
    InferenceMapping,
    configured_input_mapping_path,
    resolve_inference_mapping,
)
from retrieval_core.stages.base import StageContext
from retrieval_core.utils.artifacts import index_artifact_path
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
    inputs: dict[str, Any] = {"dataset": str(cfg.dataset.name)}
    index_parameters = _index_parameters(cfg)
    if index_parameters:
        inputs["index_id"] = str(cfg.selections.index_id)
    index_paths = [
        str(project_path(index_path))
        for parameters in index_parameters
        if (index_path := _configured_index_path(parameters))
    ]
    if len(index_paths) == 1:
        inputs["index_path"] = index_paths[0]
    elif index_paths:
        inputs["index_paths"] = index_paths
    input_mapping_path = configured_input_mapping_path(cfg)
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
    raw_query_content = query.get(EVALUATION_DATA_SCHEMA.query_content)
    query_content = None if raw_query_content is None else str(raw_query_content)
    query_meta = _query_meta(query)
    candidate_document_ids = list(inference_mapping.candidate_ids(query_input))
    result = await pipeline.run_async(
        data={
            INFERENCE_INPUT_COMPONENT: {
                "query": query_content or "",
                "query_meta": query_meta,
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
    output = result[INFERENCE_OUTPUT_COMPONENT]
    documents = list(output[INFERENCE_DOCUMENTS_FIELD])
    parsed_query_content = output.get(EVALUATION_DATA_SCHEMA.query_content)
    if parsed_query_content is None:
        parsed_query_content = query_content
    if parsed_query_content is None:
        raise ValueError(
            "The inference pipeline did not return query_content and the query record "
            f"{query_id!r} has no {EVALUATION_DATA_SCHEMA.query_content!r} field."
        )
    prediction = {
        EVALUATION_DATA_SCHEMA.query_id: query_id,
        EVALUATION_DATA_SCHEMA.IN: query_input,
        EVALUATION_DATA_SCHEMA.query_content: str(parsed_query_content),
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
    EVALUATION_DATA_SCHEMA.validate_prediction(prediction)
    return prediction


def _query_meta(query: dict[str, Any]) -> dict[str, Any]:
    """Return query fields available to parser components as metadata."""

    reserved_fields = {
        EVALUATION_DATA_SCHEMA.query_id,
        EVALUATION_DATA_SCHEMA.IN,
        "meta",
    }
    meta = {key: value for key, value in query.items() if key not in reserved_fields}
    meta.update(dict(query.get("meta") or {}))
    return meta


def prepare_inference_config(cfg: DictConfig) -> None:
    """Validate the canonical index selected by an index-backed pipeline."""

    index_parameters = _index_parameters(cfg)
    selections = cfg.get("selections")
    index_id = selections.get("index_id") if selections else None
    if not index_parameters:
        if index_id:
            raise ValueError(
                "selections.index_id is only valid for pipelines with an index_path "
                "component init parameter."
            )
        return
    if index_id is None or not str(index_id).strip():
        raise ValueError(
            "The selected inference pipeline requires a non-empty selections.index_id."
        )

    expected_path = index_artifact_path(cfg.paths.indexes_dir, str(index_id))
    configured_paths = [_configured_index_path(parameters) for parameters in index_parameters]
    for configured_path in configured_paths:
        resolved_path = project_path(configured_path)
        if resolved_path != expected_path:
            raise ValueError(
                "Every component index_path must resolve from paths.indexes_dir and "
                "selections.index_id."
            )
    if not expected_path.is_file():
        raise FileNotFoundError(
            f"No index exists with selections.index_id={index_id!r}: {expected_path}"
        )


def _index_parameters(cfg: DictConfig) -> list[DictConfig]:
    """Return component init-parameter mappings that declare an index path."""

    pipeline = cfg.get("pipeline")
    components = pipeline.get("components") if pipeline else None
    if not components:
        return []
    parameters: list[DictConfig] = []
    for component in components.values():
        init_parameters = component.get("init_parameters")
        if init_parameters is not None and "index_path" in init_parameters.keys():
            parameters.append(init_parameters)
    return parameters


def _configured_index_path(parameters: DictConfig) -> Any:
    """Read an index path without resolving OmegaConf's mandatory-value sentinel."""

    if OmegaConf.is_missing(parameters, "index_path"):
        return None
    return parameters.get("index_path")
