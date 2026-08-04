"""Indexing stage runner."""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path
from typing import Protocol, runtime_checkable

from haystack import Document
from omegaconf import DictConfig
from tqdm import tqdm

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.stages.base import Stage
from retrieval_core.utils.artifacts import index_artifact_path
from retrieval_core.utils.io import project_path
from retrieval_core.utils.pipelines import (
    load_async_pipeline,
    without_component_progress_bars,
)

INDEXING_INPUT_COMPONENT = "input"
INDEXING_WRITER_COMPONENT = "indexer"
INDEXING_OUTPUT_COMPONENT = "output"

logger = logging.getLogger(__name__)


@runtime_checkable
class IndexWriter(Protocol):
    """Lifecycle required from the reserved indexing component."""

    def write_session(self) -> AbstractAsyncContextManager[None]: ...


class IndexingStage(Stage):
    """Build and publish one immutable retrieval index."""

    def prepare_config(self) -> None:
        index_id = str(self.cfg.selections.index_id)
        canonical_index_path = index_artifact_path(self.cfg.paths.indexes_dir, index_id)
        configured_output_path = _configured_indexer_output_path(self.cfg)
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

        batch_size = int(self.cfg.runtime.indexing_batch_size)
        if batch_size <= 0:
            raise ValueError("runtime.indexing_batch_size must be greater than zero.")

    async def run(self) -> dict:
        cfg = self.cfg
        index_id = str(cfg.selections.index_id)
        canonical_index_path = index_artifact_path(cfg.paths.indexes_dir, index_id)
        batch_size = int(cfg.runtime.indexing_batch_size)
        documents_path = project_path(cfg.dataset.documents_path)
        logger.info(
            "Indexing documents: dataset=%s documents_path=%s index_id=%s batch_size=%d",
            cfg.dataset.name,
            documents_path,
            index_id,
            batch_size,
        )
        pipeline_config = without_component_progress_bars(cfg.pipeline)
        pipeline = load_async_pipeline(pipeline_config)
        index_writer = pipeline.get_component(INDEXING_WRITER_COMPONENT)
        if not isinstance(index_writer, IndexWriter):
            raise TypeError(
                f"The reserved {INDEXING_WRITER_COMPONENT!r} component must implement IndexWriter."
            )
        documents_sha256 = hashlib.sha256()
        document_ids: set[str] = set()
        document_count = 0
        indexed_count = 0
        batch_count = 0
        batch: list[Document] = []
        reserved_fields = {
            EVALUATION_DATA_SCHEMA.doc_id,
            EVALUATION_DATA_SCHEMA.text,
            "meta",
            "score",
            "embedding",
        }

        async def index_batch(documents: list[Document]) -> int:
            result = await pipeline.run_async(
                data={
                    INDEXING_INPUT_COMPONENT: {"documents": documents},
                },
                include_outputs_from={INDEXING_OUTPUT_COMPONENT},
                concurrency_limit=int(cfg.runtime.concurrency_limit),
            )
            output = result[INDEXING_OUTPUT_COMPONENT]
            if project_path(output["index_path"]) != canonical_index_path:
                raise RuntimeError("Indexing pipeline returned an unexpected index path.")
            count = output["indexed_count"]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise RuntimeError(f"Indexing pipeline returned invalid indexed_count: {count!r}")
            return count

        source_document_total = _count_nonempty_lines(documents_path)
        show_progress = bool(cfg.runtime.get("progress_bar", True))
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(index_writer.write_session())
            progress = stack.enter_context(
                tqdm(
                    total=source_document_total,
                    desc="Indexing",
                    unit="doc",
                    disable=not show_progress,
                )
            )
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
                        key: value for key, value in record.items() if key not in reserved_fields
                    }
                    meta.update(dict(record.get("meta") or {}))
                    batch.append(
                        Document(
                            id=document_id,
                            content=str(record[EVALUATION_DATA_SCHEMA.text]),
                            meta=meta,
                            score=record.get("score"),
                            embedding=record.get("embedding"),
                        )
                    )
                    document_count += 1

                    if len(batch) < batch_size:
                        continue
                    completed_batch_size = len(batch)
                    indexed_count += await index_batch(batch)
                    batch_count += 1
                    progress.update(completed_batch_size)
                    logger.debug(
                        "Indexed batch: batch=%d source_documents=%d indexed_documents=%d",
                        batch_count,
                        document_count,
                        indexed_count,
                    )
                    batch = []

            if batch:
                completed_batch_size = len(batch)
                indexed_count += await index_batch(batch)
                batch_count += 1
                progress.update(completed_batch_size)
                logger.debug(
                    "Indexed final batch: batch=%d source_documents=%d indexed_documents=%d",
                    batch_count,
                    document_count,
                    indexed_count,
                )
        result = {
            INDEXING_OUTPUT_COMPONENT: {
                "index_path": str(canonical_index_path),
                "indexed_count": indexed_count,
            },
            "source_document_count": document_count,
            "batch_count": batch_count,
        }

        self.write_result(result)
        self.write_manifest(
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
        logger.info(
            "Index published: index_path=%s source_documents=%d indexed_documents=%d batches=%d",
            canonical_index_path,
            document_count,
            indexed_count,
            batch_count,
        )
        return result


def _configured_indexer_output_path(cfg: DictConfig) -> str:
    """Return the configured path of the reserved index-writer component."""

    components = cfg.pipeline.get("components")
    if not components or INDEXING_WRITER_COMPONENT not in components:
        raise ValueError(
            f"The indexing pipeline must define the reserved "
            f"{INDEXING_WRITER_COMPONENT!r} component."
        )
    init_parameters = components[INDEXING_WRITER_COMPONENT].get("init_parameters")
    if init_parameters is None or not init_parameters.get("output_path"):
        raise ValueError(
            f"The reserved {INDEXING_WRITER_COMPONENT!r} component must declare output_path."
        )
    return str(init_parameters.output_path)


def _count_nonempty_lines(path: Path) -> int:
    """Count source documents for one stage-level progress total."""

    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)
