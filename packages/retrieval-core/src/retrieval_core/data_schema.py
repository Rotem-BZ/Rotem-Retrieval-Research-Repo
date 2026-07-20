"""Canonical field names for evaluation datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationDataSchema:
    """Required JSONL fields shared by queries, documents, and qrels.

    `IN` is the query-to-qrel join key; `query_id` is the external query identifier.
    """

    query_id: str = "query_id"
    IN: str = "IN"
    query_content: str = "query_content"
    doc_id: str = "doc_id"
    text: str = "text"
    label: str = "label"

    def validate_query(self, record: Mapping[str, Any]) -> None:
        self._validate(record, (self.query_id, self.IN, self.query_content), "Query")

    def validate_document(self, record: Mapping[str, Any]) -> None:
        self._validate(record, (self.doc_id, self.text), "Document")

    def validate_qrel(self, record: Mapping[str, Any]) -> None:
        self._validate(record, (self.IN, self.doc_id, self.label), "Qrel")

    @staticmethod
    def _validate(
        record: Mapping[str, Any],
        required_fields: tuple[str, ...],
        record_type: str,
    ) -> None:
        missing_fields = [field for field in required_fields if field not in record]
        if missing_fields:
            raise ValueError(f"{record_type} record is missing required fields: {missing_fields}")


EVALUATION_DATA_SCHEMA = EvaluationDataSchema()
