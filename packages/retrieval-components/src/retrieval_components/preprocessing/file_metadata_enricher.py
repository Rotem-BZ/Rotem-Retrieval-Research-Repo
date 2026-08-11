"""Enrich queries and documents from an in-memory JSONL metadata mapping."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from haystack import Document, component

from retrieval_components.dataclasses.query import Query


def _validate_mapping_path(mapping_path: str) -> Path:
    if not isinstance(mapping_path, str) or not mapping_path:
        raise ValueError("mapping_path must be a non-empty string.")
    path = Path(mapping_path)
    if not path.is_absolute():
        raise ValueError(f"mapping_path must be absolute: {mapping_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Metadata mapping file does not exist: {path}")
    return path


def _load_metadata_mapping(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in metadata mapping {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise TypeError(
                    f"Metadata mapping {path} line {line_number} must be a JSON object."
                )
            record_id = record.get("id")
            metadata = record.get("meta")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"Metadata mapping {path} line {line_number} requires a non-empty string `id`."
                )
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Metadata mapping {path} line {line_number} requires an object `meta`."
                )
            unexpected = set(record) - {"id", "meta"}
            if unexpected:
                raise ValueError(
                    f"Metadata mapping {path} line {line_number} has unexpected fields: "
                    f"{sorted(unexpected)}."
                )
            if record_id in mapping:
                raise ValueError(
                    f"Metadata mapping {path} contains duplicate id {record_id!r} "
                    f"at line {line_number}."
                )
            mapping[record_id] = metadata
    return mapping


def _merged_metadata(
    existing: dict[str, Any], mapped: dict[str, Any], *, owner: str
) -> dict[str, Any]:
    conflicts = set(existing) & set(mapped)
    if conflicts:
        raise ValueError(f"{owner} metadata conflicts on keys: {sorted(conflicts)}.")
    merged = deepcopy(existing)
    merged.update(deepcopy(mapped))
    return merged


class _MetadataMappingMixin:
    def __init__(self, mapping_path: str) -> None:
        self.mapping_path = mapping_path
        self._mapping_file = _validate_mapping_path(mapping_path)
        self._metadata_by_id: dict[str, dict[str, Any]] | None = None

    def warm_up(self) -> None:
        """Load and validate the complete mapping exactly once."""

        if self._metadata_by_id is None:
            self._metadata_by_id = _load_metadata_mapping(self._mapping_file)

    def _metadata_for(self, record_id: str, *, owner: str) -> dict[str, Any]:
        self.warm_up()
        assert self._metadata_by_id is not None
        try:
            return self._metadata_by_id[record_id]
        except KeyError as error:
            raise ValueError(f"{owner} has no metadata mapping for id {record_id!r}.") from error


@component
class QueryMetadataEnricher(_MetadataMappingMixin):
    """Attach mapped metadata to an immutable Query."""

    def __init__(self, mapping_path: str) -> None:
        _MetadataMappingMixin.__init__(self, mapping_path)

    @component.output_types(query=Query)
    def run(self, query: Query) -> dict[str, Query]:
        mapped = self._metadata_for(query.id, owner=f"Query {query.id!r}")
        return {
            "query": Query(
                id=query.id,
                content=query.content,
                meta=_merged_metadata(query.meta, mapped, owner=f"Query {query.id!r}"),
            )
        }


@component
class DocumentMetadataEnricher(_MetadataMappingMixin):
    """Attach source-document metadata to Haystack Documents."""

    def __init__(self, mapping_path: str) -> None:
        _MetadataMappingMixin.__init__(self, mapping_path)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        enriched: list[Document] = []
        for document in documents:
            source_id = document.meta.get("source_document_id")
            if source_id is not None and (not isinstance(source_id, str) or not source_id):
                raise ValueError(
                    f"Document {document.id!r} has an invalid `meta.source_document_id`."
                )
            lookup_id = source_id or document.id
            if not isinstance(lookup_id, str) or not lookup_id:
                raise ValueError(
                    "DocumentMetadataEnricher requires every document to define an id or "
                    "`meta.source_document_id`."
                )
            mapped = self._metadata_for(lookup_id, owner=f"Document {document.id!r}")
            document_data = document.to_dict(flatten=False)
            document_data["meta"] = _merged_metadata(
                document.meta,
                mapped,
                owner=f"Document {document.id!r}",
            )
            enriched.append(Document.from_dict(document_data))
        return {"documents": enriched}
