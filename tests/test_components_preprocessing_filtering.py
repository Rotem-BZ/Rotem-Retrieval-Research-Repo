from haystack import Document

from retrieval_research.components.filtering import DocumentContentFilter
from retrieval_research.components.preprocessing import DocumentTextPrefixer, TextPreprocessor


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
