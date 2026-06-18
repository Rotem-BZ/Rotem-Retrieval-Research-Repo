"""Inference stage runner."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from haystack import Document
from omegaconf import DictConfig
from tqdm import tqdm

from retrieval_research.input_mapping import resolve_inference_mapping
from retrieval_research.io import write_predictions
from retrieval_research.pipelines import include_outputs, load_async_pipeline, to_container
from retrieval_research.stages.base import StageContext, is_dry_run


async def run_inference(cfg: DictConfig) -> list[dict[str, Any]]:
    pipeline = load_async_pipeline(cfg.pipeline)
    context = StageContext.from_config(cfg)

    predictions: list[dict[str, Any]] = []
    inference_mapping = resolve_inference_mapping(cfg)

    for query in tqdm(inference_mapping.queries, desc="queries"):
        query_id = str(query["id"])
        inputs = _build_query_inputs(
            cfg,
            query["text"],
            candidate_document_ids=inference_mapping.candidate_ids(query_id),
            candidate_documents=inference_mapping.candidate_documents(query_id),
        )
        result = await pipeline.run_async(
            data=inputs,
            include_outputs_from=include_outputs(cfg.pipeline_run.include_outputs_from),
            concurrency_limit=int(cfg.pipeline_run.concurrency_limit),
        )
        documents = _extract_documents(result, cfg.pipeline_run.documents_output)
        predictions.append(
            {
                "query_id": query_id,
                "query": query["text"],
                "documents": [_document_to_dict(document) for document in documents],
            }
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
    return predictions


def _build_query_inputs(
    cfg: DictConfig,
    query_text: str,
    *,
    candidate_document_ids: list[str] | None = None,
    candidate_documents: list[Document] | None = None,
) -> dict[str, Any]:
    candidate_document_ids = [] if candidate_document_ids is None else candidate_document_ids
    candidate_documents = [] if candidate_documents is None else candidate_documents
    inputs = deepcopy(to_container(cfg.pipeline_run.inputs) or {})

    query_inputs = cfg.pipeline_run.get("query_inputs")
    if query_inputs is None:
        query_inputs = [cfg.pipeline_run.query_input]

    for query_input in query_inputs:
        component_name = str(query_input.component)
        parameter_name = str(query_input.parameter)
        component_inputs = inputs.setdefault(component_name, {})
        component_inputs[parameter_name] = query_text

    for candidate_id_input in _pipeline_run_entries(cfg, "candidate_id_inputs", "candidate_id_input"):
        component_name = str(candidate_id_input.component)
        parameter_name = str(candidate_id_input.parameter)
        component_inputs = inputs.setdefault(component_name, {})
        component_inputs[parameter_name] = list(candidate_document_ids)

    for candidate_document_input in _pipeline_run_entries(cfg, "candidate_document_inputs"):
        component_name = str(candidate_document_input.component)
        parameter_name = str(candidate_document_input.parameter)
        component_inputs = inputs.setdefault(component_name, {})
        component_inputs[parameter_name] = list(candidate_documents)

    return inputs


def _pipeline_run_entries(
    cfg: DictConfig,
    plural_key: str,
    singular_key: str | None = None,
) -> list[Any]:
    entries = cfg.pipeline_run.get(plural_key)
    if entries is None and singular_key is not None:
        singular_entry = cfg.pipeline_run.get(singular_key)
        entries = [] if singular_entry is None else [singular_entry]
    return list(entries or [])


def _extract_documents(result: dict[str, Any], output_config: DictConfig) -> list[Document]:
    component_name = str(output_config.component)
    field_name = str(output_config.field)
    documents = result[component_name][field_name]
    return list(documents)


def _document_to_dict(document: Document) -> dict[str, Any]:
    return {
        "id": document.id,
        "content": document.content,
        "meta": dict(document.meta or {}),
        "score": getattr(document, "score", None),
    }
