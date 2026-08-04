import pytest
from haystack import Document

from retrieval_components.dataclasses.query import Query
from retrieval_components.filtering.document_content_filter import DocumentContentFilter
from retrieval_components.preprocessing.document_text_prefixer import DocumentTextPrefixer
from retrieval_components.preprocessing.identity_parser import IdentityParser
from retrieval_components.preprocessing.query_text_preprocessor import QueryTextPreprocessor
from retrieval_components.preprocessing.query_to_string import QueryToString


def test_document_text_prefixer_preserves_metadata() -> None:
    prefixer = DocumentTextPrefixer(prefix="passage: ", replace_regexes={"bad": "good"})
    result = prefixer.run([Document(id="d1", content="bad text", meta={"source": "toy"})])

    assert result["documents"][0].content == "passage: good text"
    assert result["documents"][0].meta == {"source": "toy"}


@pytest.mark.parametrize("component", [DocumentTextPrefixer(), DocumentContentFilter()])
def test_document_text_components_reject_missing_content(component) -> None:
    with pytest.raises(ValueError, match="content"):
        component.run([Document(id="d1")])


def test_identity_parser_returns_documents_unchanged() -> None:
    parser = IdentityParser()
    documents = [
        Document(
            id="d1",
            content="already materialized",
            meta={"nested": {"language": "en"}},
            score=0.5,
            embedding=[1.0, 2.0],
        )
    ]

    parsed = parser.run(documents)["documents"]

    assert parsed is documents
    assert parsed[0] is documents[0]


def test_identity_parser_rejects_missing_content() -> None:
    with pytest.raises(ValueError, match="Document 'd1' is missing content"):
        IdentityParser().run([Document(id="d1")])


def test_query_preprocessor_and_to_string_preserve_query_fields() -> None:
    source = Query(
        id="q1",
        content="  HYDRA\nCONFIG  ",
        meta={"nested": {"language": "en"}},
    )
    preprocessor = QueryTextPreprocessor(
        prefix="query: ",
        lowercase=True,
        replace_regexes={r"\s+": " "},
    )

    transformed = preprocessor.run(source)["query"]

    assert transformed.content == "query: hydra config"
    assert transformed.id == "q1"
    assert transformed.meta == source.meta
    assert QueryToString().run(transformed) == {"text": "query: hydra config"}


def test_document_content_filter_uses_regex_and_word_bounds() -> None:
    content_filter = DocumentContentFilter(include_regex="retrieval", min_words=2, max_words=4)
    result = content_filter.run(
        [
            Document(id="keep", content="retrieval works well"),
            Document(id="too-short", content="retrieval"),
            Document(id="no-match", content="configuration works well"),
        ]
    )

    assert [document.id for document in result["documents"]] == ["keep"]
    assert [document.id for document in result["rejected_documents"]] == [
        "too-short",
        "no-match",
    ]
