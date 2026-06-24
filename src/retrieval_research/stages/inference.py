"""Inference stage runner."""

from __future__ import annotations

from typing import Any

from haystack import Document
from omegaconf import DictConfig
from tqdm import tqdm

from retrieval_research.input_mapping import resolve_inference_mapping
from retrieval_research.io import write_predictions
from retrieval_research.pipelines import load_async_pipeline
from retrieval_research.stages.base import StageContext, is_dry_run

INFERENCE_INPUT_COMPONENT = "input"
INFERENCE_OUTPUT_COMPONENT = "output"
INFERENCE_DOCUMENTS_FIELD = "documents"


async def run_inference(cfg: DictConfig) -> list[dict[str, Any]]:
    pipeline = load_async_pipeline(cfg.pipeline)
    context = StageContext.from_config(cfg)

    predictions: list[dict[str, Any]] = []
    inference_mapping = resolve_inference_mapping(cfg)

    for query in tqdm(inference_mapping.queries, desc="queries"):
        query_id = str(query["id"])
        inputs = _build_query_inputs(
            query["text"],
            candidate_document_ids=inference_mapping.candidate_ids(query_id),
            candidate_documents=inference_mapping.candidate_documents(query_id),
        )
        result = await pipeline.run_async(
            data=inputs,
            include_outputs_from={INFERENCE_OUTPUT_COMPONENT},
            concurrency_limit=int(cfg.runtime.concurrency_limit),
        )
        documents = _extract_documents(result)
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
