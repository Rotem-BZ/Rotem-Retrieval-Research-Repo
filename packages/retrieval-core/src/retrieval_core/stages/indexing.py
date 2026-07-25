"""Indexing stage runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from haystack import Document
from omegaconf import DictConfig

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.stages.base import StageContext
from retrieval_core.utils.artifacts import index_artifact_path
from retrieval_core.utils.io import project_path
from retrieval_core.utils.pipelines import load_async_pipeline

INDEXING_INPUT_COMPONENT = "input"
INDEXING_OUTPUT_COMPONENT = "output"


async def run_indexing(cfg: DictConfig) -> dict:
    index_id = str(cfg.selections.index_id)
    canonical_index_path = index_artifact_path(cfg.paths.indexes_dir, index_id)
    indexer_name, configured_output_path = _configured_indexer(cfg)
    configured_index_path = project_path(configured_output_path)
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

    batch_size = int(cfg.runtime.indexing_batch_size)
    if batch_size <= 0:
        raise ValueError("runtime.indexing_batch_size must be greater than zero.")

    documents_path = project_path(cfg.dataset.documents_path)
    pipeline = load_async_pipeline(cfg.pipeline)
    indexer = pipeline.get_component(indexer_name)
    begin_batch_write = getattr(indexer, "begin_batch_write", None)
    finish_batch_write = getattr(indexer, "finish_batch_write", None)
    abort_batch_write = getattr(indexer, "abort_batch_write", None)
    if not all(
        callable(method)
        for method in (begin_batch_write, finish_batch_write, abort_batch_write)
    ):
        raise TypeError(
            f"Indexing component {indexer_name!r} must implement begin_batch_write(), "
            "finish_batch_write(), and abort_batch_write() for staged batch indexing."
        )

    context = StageContext.from_config(cfg)
    canonical_index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index_path = canonical_index_path.with_name(
        f".{canonical_index_path.name}.{uuid4().hex}.tmp"
    )

    documents_sha256 = hashlib.sha256()
    document_ids: set[str] = set()
    document_count = 0
    indexed_count = 0
    batch_count = 0
    batch: list[Document] = []
    reserved_fields = {
        EVALUATION_DATA_SCHEMA.doc_id,
        "meta",
        "score",
        "embedding",
    }
    try:
        begin_batch_write(str(temporary_index_path))
        with documents_path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                documents_sha256.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in documents file at line {line_number}: "
                        f"{documents_path}"
                    ) from error
                EVALUATION_DATA_SCHEMA.validate_document(record)
                document_id = str(record[EVALUATION_DATA_SCHEMA.doc_id])
                if document_id in document_ids:
                    raise ValueError(f"Duplicate document id in dataset: {document_id}")
                document_ids.add(document_id)

                meta = {
                    key: value
                    for key, value in record.items()
                    if key not in reserved_fields
                }
                meta.update(dict(record.get("meta") or {}))
                batch.append(
                    Document(
                        id=document_id,
                        meta=meta,
                        score=record.get("score"),
                        embedding=record.get("embedding"),
                    )
                )
                document_count += 1

                if len(batch) < batch_size:
                    continue
                batch_result = await pipeline.run_async(
                    data={INDEXING_INPUT_COMPONENT: {"documents": batch}},
                    include_outputs_from={INDEXING_OUTPUT_COMPONENT},
                    concurrency_limit=int(cfg.runtime.concurrency_limit),
                )
                indexed_count += _validate_batch_result(
                    batch_result,
                    temporary_index_path=temporary_index_path,
                )
                batch_count += 1
                batch = []

        if batch:
            batch_result = await pipeline.run_async(
                data={INDEXING_INPUT_COMPONENT: {"documents": batch}},
                include_outputs_from={INDEXING_OUTPUT_COMPONENT},
                concurrency_limit=int(cfg.runtime.concurrency_limit),
            )
            indexed_count += _validate_batch_result(
                batch_result,
                temporary_index_path=temporary_index_path,
            )
            batch_count += 1

        finish_batch_write()
        temporary_index_path.replace(canonical_index_path)
    except BaseException:
        abort_batch_write()
        temporary_index_path.unlink(missing_ok=True)
        raise

    result = {
        INDEXING_OUTPUT_COMPONENT: {
            "index_path": str(canonical_index_path),
            "indexed_count": indexed_count,
        },
        "source_document_count": document_count,
        "batch_count": batch_count,
    }

    context.write_resolved_config()
    context.write_result(result)
    context.write_manifest(
        artifacts={"index": canonical_index_path},
        inputs={
            "index_id": index_id,
            "dataset": str(cfg.dataset.name),
            "documents_path": str(documents_path),
            "documents_sha256": documents_sha256.hexdigest(),
            "document_count": document_count,
            "batch_count": batch_count,
        },
    )
    return result


def _configured_indexer(cfg: DictConfig) -> tuple[str, str]:
    """Return the name and output path of the component producing the index."""

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
    return component_name, str(init_parameters.output_path)


def _validate_batch_result(
    result: dict[str, Any],
    *,
    temporary_index_path: Path,
) -> int:
    """Validate one pipeline batch result and return its indexed count."""

    output = result.get(INDEXING_OUTPUT_COMPONENT)
    if not isinstance(output, dict):
        raise RuntimeError("Indexing pipeline batch did not return output.")
    index_path = output.get("index_path")
    if not index_path or project_path(index_path) != temporary_index_path:
        raise RuntimeError(
            "Indexing pipeline batch returned an unexpected temporary index path: "
            f"{index_path!r}"
        )
    indexed_count = output.get("indexed_count")
    if (
        not isinstance(indexed_count, int)
        or isinstance(indexed_count, bool)
        or indexed_count < 0
    ):
        raise RuntimeError(
            "Indexing pipeline batch returned an invalid indexed_count: "
            f"{indexed_count!r}"
        )
    return indexed_count
