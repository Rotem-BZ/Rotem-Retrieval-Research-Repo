import pytest
from haystack import Document

from retrieval_components import Query
from retrieval_components.filtering import DocumentContentFilter
from retrieval_components.preprocessing import (
    DocumentContentFieldParser,
    DocumentTextPrefixer,
    QueryContentAdapter,
    QueryContentFieldParser,
    QueryTextPreprocessor,
    TextPreprocessor,
)


def test_text_preprocessor_cleans_and_prefixes_query() -> None:
    preprocessor = TextPreprocessor(
        prefix="query: ",
        lowercase=True,
        replace_regexes={r"\s+": " "},
    )

    assert preprocessor.run("  HYDRA\nCONFIG  ") == {"text": "query: hydra config"}


def test_document_text_prefixer_preserves_metadata() -> None:
    prefixer = DocumentTextPrefixer(prefix="passage: ", replace_regexes={"bad": "good"})
    result = prefixer.run([Document(id="d1", content="bad text", meta={"source": "toy"})])

    assert result["documents"][0].content == "passage: good text"
    assert result["documents"][0].meta == {"source": "toy"}


def test_document_content_field_parser_sets_content_and_preserves_document_fields() -> None:
    parser = DocumentContentFieldParser(content_field="body")
    source = Document(
        id="d1",
        content=None,
        meta={"body": "parsed text", "nested": {"language": "en"}},
        score=0.5,
        embedding=[1.0, 2.0],
    )

    parsed = parser.run([source])["documents"][0]

    assert parsed.content == "parsed text"
    assert parsed.meta == source.meta
    assert parsed.score == 0.5
    assert parsed.embedding == [1.0, 2.0]


def test_content_field_parsers_report_missing_configured_fields() -> None:
    with pytest.raises(ValueError, match="Document 'd1'.*'body'"):
        DocumentContentFieldParser(content_field="body").run([Document(id="d1")])

    with pytest.raises(ValueError, match="Query.*'question'"):
        QueryContentFieldParser(content_field="question").run(
            Query(id="q1", meta={"language": "en"})
        )


def test_query_content_field_parser_uses_nested_json_value() -> None:
    parser = QueryContentFieldParser(content_field="question")
    source = Query(id="q1", meta={"question": {"text": "nested"}})

    parsed = parser.run(source)["query"]

    assert parsed.content == "{'text': 'nested'}"
    assert parsed.meta == source.meta
    assert source.content is None


def test_query_preprocessor_and_adapter_preserve_query_fields() -> None:
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
    assert QueryContentAdapter().run(transformed) == {"text": "query: hydra config"}


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
