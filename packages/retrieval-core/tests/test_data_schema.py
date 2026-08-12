from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from retrieval_core.data_schema import Qrel, document_from_dict, query_from_dict


def test_qrel_is_an_immutable_round_trippable_dataclass() -> None:
    qrel = Qrel(IN="need-1", document_id="doc-1", label=2)
    assert is_dataclass(qrel)
    assert qrel.to_dict() == {"IN": "need-1", "document_id": "doc-1", "label": 2}
    assert Qrel.from_dict(qrel.to_dict()) == qrel
    with pytest.raises(FrozenInstanceError):
        qrel.label = 1  # type: ignore[misc]


def test_canonical_query_and_document_records_materialize_dataclasses() -> None:
    query = query_from_dict(
        {"id": "query-1", "content": "question", "IN": "need-1", "meta": {}}
    )
    document = document_from_dict(
        {"id": "doc-1", "content": "answer", "meta": {"source": "test"}}
    )
    assert query.id == "query-1"
    assert query.IN == "need-1"
    assert document.id == "doc-1"
    assert document.meta == {"source": "test"}


def test_document_loader_accepts_document_fields_and_applies_defaults() -> None:
    document = document_from_dict(
        {
            "content": "answer",
            "score": 0.75,
            "embedding": [0.1, 0.2],
        }
    )

    assert document.id
    assert document.meta == {}
    assert document.score == 0.75
    assert document.embedding == [0.1, 0.2]


@pytest.mark.parametrize(
    ("loader", "record", "message"),
    [
        (query_from_dict, {"id": "q", "content": "x", "IN": "n"}, "meta"),
        (document_from_dict, {"id": "d", "meta": {}}, "content"),
        (Qrel.from_dict, {"IN": "n", "document_id": "d"}, "label"),
    ],
)
def test_canonical_records_require_all_fields(loader, record, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        loader(record)


def test_canonical_records_reject_unknown_top_level_fields() -> None:
    with pytest.raises(ValueError, match="unexpected fields.*language"):
        query_from_dict(
            {
                "id": "q",
                "content": "x",
                "IN": "n",
                "meta": {},
                "language": "en",
            }
        )


@pytest.mark.parametrize(
    "qrel",
    [
        {"IN": "", "document_id": "d", "label": 1},
        {"IN": "n", "document_id": "", "label": 1},
    ],
)
def test_qrel_requires_nonempty_identifiers(qrel: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Qrel.from_dict(qrel)


def test_qrel_requires_an_integer_label() -> None:
    with pytest.raises(TypeError, match="integer"):
        Qrel.from_dict({"IN": "n", "document_id": "d", "label": 1.5})
