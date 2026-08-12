import json
import logging
from pathlib import Path

import pytest
from haystack import Document

from retrieval_components.dataclasses.query import Query
from retrieval_components.preprocessing.file_metadata_enricher import (
    DocumentMetadataEnricher,
    QueryMetadataEnricher,
)


def _mapping_file(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "metadata.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_query_enricher_preserves_nested_metadata_and_input(tmp_path: Path) -> None:
    path = _mapping_file(
        tmp_path,
        [{"id": "q1", "meta": {"source": {"site": "example"}, "labels": ["a"]}}],
    )
    query = Query(id="q1", content="text", meta={"split": "test"})

    enriched = QueryMetadataEnricher(str(path)).run(query)["query"]

    assert enriched == Query(
        id="q1",
        content="text",
        meta={"split": "test", "source": {"site": "example"}, "labels": ["a"]},
    )
    assert query.meta == {"split": "test"}
    assert enriched is not query


def test_document_enricher_uses_source_id_and_preserves_fields(tmp_path: Path) -> None:
    path = _mapping_file(tmp_path, [{"id": "d1", "meta": {"nested": {"value": 1}}}])
    document = Document(
        id="d1::chunk-0",
        content="chunk",
        meta={"source_document_id": "d1", "chunk_index": 0},
        score=0.75,
        embedding=[1.0, 2.0],
    )

    enriched = DocumentMetadataEnricher(str(path)).run([document])["documents"][0]

    assert enriched.id == document.id
    assert enriched.content == document.content
    assert enriched.score == document.score
    assert enriched.embedding == document.embedding
    assert enriched.meta == {
        "source_document_id": "d1",
        "chunk_index": 0,
        "nested": {"value": 1},
    }
    assert document.meta == {"source_document_id": "d1", "chunk_index": 0}
    assert enriched is not document


def test_document_enricher_accepts_an_empty_batch(tmp_path: Path) -> None:
    path = _mapping_file(tmp_path, [])
    assert DocumentMetadataEnricher(str(path)).run([]) == {"documents": []}


@pytest.mark.parametrize("source_id", ["", 3, ["d1"]])
def test_document_enricher_rejects_invalid_source_id(tmp_path: Path, source_id: object) -> None:
    path = _mapping_file(tmp_path, [{"id": "d1", "meta": {"topic": "x"}}])
    document = Document(id="d1", meta={"source_document_id": source_id})
    with pytest.raises(ValueError, match="invalid `meta.source_document_id`"):
        DocumentMetadataEnricher(str(path)).run([document])


def test_enrichers_require_matching_ids(tmp_path: Path) -> None:
    path = _mapping_file(tmp_path, [{"id": "other", "meta": {}}])
    with pytest.raises(ValueError, match="no metadata mapping"):
        QueryMetadataEnricher(str(path)).run(Query(id="q1"))
    with pytest.raises(ValueError, match="no metadata mapping"):
        DocumentMetadataEnricher(str(path)).run([Document(id="d1")])


def test_enrichers_reject_top_level_conflicts(tmp_path: Path) -> None:
    path = _mapping_file(tmp_path, [{"id": "q1", "meta": {"nested": {"new": 2}}}])
    with pytest.raises(ValueError, match="conflicts on keys.*nested"):
        QueryMetadataEnricher(str(path)).run(
            Query(id="q1", meta={"nested": {"existing": 1}})
        )


def test_warm_up_is_idempotent_and_does_not_reload_changes(tmp_path: Path) -> None:
    path = _mapping_file(tmp_path, [{"id": "q1", "meta": {"version": 1}}])
    enricher = QueryMetadataEnricher(str(path))
    enricher.warm_up()
    _mapping_file(tmp_path, [{"id": "q1", "meta": {"version": 2}}])
    enricher.warm_up()
    assert enricher.run(Query(id="q1"))["query"].meta == {"version": 1}


def test_warm_up_logs_load_and_reuse_at_debug_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _mapping_file(
        tmp_path,
        [
            {"id": "q1", "meta": {}},
            {"id": "q2", "meta": {}},
        ],
    )
    enricher = QueryMetadataEnricher(str(path))

    with caplog.at_level(
        logging.DEBUG,
        logger="retrieval_components.preprocessing.file_metadata_enricher",
    ):
        enricher.warm_up()
        enricher.warm_up()

    assert f"Loading metadata mapping from {path}." in caplog.messages
    assert f"Loaded 2 metadata records from {path}." in caplog.messages
    assert f"Metadata mapping is already loaded from {path} with 2 records." in caplog.messages


def test_mapping_allows_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text('\n{"id": "q1", "meta": {}}\n\n', encoding="utf-8")
    assert QueryMetadataEnricher(str(path)).run(Query(id="q1"))["query"].meta == {}


@pytest.mark.parametrize(
    ("contents", "exception_type", "message"),
    [
        ("not json\n", ValueError, "Invalid JSON"),
        ('[]\n', TypeError, "must be a JSON object"),
        ('{"id": "", "meta": {}}\n', ValueError, "non-empty string `id`"),
        ('{"id": "q1", "meta": []}\n', TypeError, "object `meta`"),
        ('{"id": "q1", "meta": {}, "extra": 1}\n', ValueError, "unexpected fields"),
        (
            '{"id": "q1", "meta": {}}\n{"id": "q1", "meta": {}}\n',
            ValueError,
            "duplicate id",
        ),
    ],
)
def test_mapping_validation(
    tmp_path: Path, contents: str, exception_type: type[Exception], message: str
) -> None:
    path = tmp_path / "metadata.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(exception_type, match=message):
        QueryMetadataEnricher(str(path)).warm_up()


def test_constructor_requires_absolute_existing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        QueryMetadataEnricher("metadata.jsonl")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        QueryMetadataEnricher(str(tmp_path / "missing.jsonl"))
