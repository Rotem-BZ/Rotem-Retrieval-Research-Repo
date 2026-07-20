from dataclasses import is_dataclass

import pytest

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA, EvaluationDataSchema


def test_evaluation_data_schema_is_an_immutable_dataclass() -> None:
    assert is_dataclass(EvaluationDataSchema)
    assert EVALUATION_DATA_SCHEMA == EvaluationDataSchema()
    assert EVALUATION_DATA_SCHEMA.query_id == "query_id"
    assert EVALUATION_DATA_SCHEMA.IN == "IN"
    assert EVALUATION_DATA_SCHEMA.query_content == "query_content"
    assert EVALUATION_DATA_SCHEMA.doc_id == "doc_id"
    assert EVALUATION_DATA_SCHEMA.text == "text"
    assert EVALUATION_DATA_SCHEMA.label == "label"


def test_schema_validation_allows_additional_fields() -> None:
    EVALUATION_DATA_SCHEMA.validate_query(
        {
            EVALUATION_DATA_SCHEMA.query_id: "query-1",
            EVALUATION_DATA_SCHEMA.IN: "1",
            EVALUATION_DATA_SCHEMA.query_content: "content",
            "language": "en",
        }
    )
    EVALUATION_DATA_SCHEMA.validate_document(
        {
            EVALUATION_DATA_SCHEMA.doc_id: "doc-1",
            EVALUATION_DATA_SCHEMA.text: "text",
            "title": "Optional title",
        }
    )
    EVALUATION_DATA_SCHEMA.validate_qrel(
        {
            EVALUATION_DATA_SCHEMA.IN: "1",
            EVALUATION_DATA_SCHEMA.doc_id: "doc-1",
            EVALUATION_DATA_SCHEMA.label: 2,
            "annotator": "test",
        }
    )


def test_schema_validation_reports_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="query_content"):
        EVALUATION_DATA_SCHEMA.validate_query(
            {
                EVALUATION_DATA_SCHEMA.query_id: "query-1",
                EVALUATION_DATA_SCHEMA.IN: "1",
            }
        )

    with pytest.raises(ValueError, match="text"):
        EVALUATION_DATA_SCHEMA.validate_document({EVALUATION_DATA_SCHEMA.doc_id: "doc-1"})

    with pytest.raises(ValueError, match="label"):
        EVALUATION_DATA_SCHEMA.validate_qrel(
            {
                EVALUATION_DATA_SCHEMA.IN: "1",
                EVALUATION_DATA_SCHEMA.doc_id: "doc-1",
            }
        )
