from haystack import Document

from retrieval_components.chunking import LangChainDocumentSplitter
from retrieval_components.chunking import langchain_document_splitter as splitter_module
from retrieval_components.dataclasses import Query
from retrieval_components.reformulation import HttpQueryReformulator


def test_langchain_document_splitter_uses_recursive_splitter_and_drops_empty_chunks(
    monkeypatch,
) -> None:
    calls = []

    class FakeSplitter:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def split_text(self, text: str) -> list[str]:
            return ["alpha", "   ", "beta"]

    monkeypatch.setattr(splitter_module, "RecursiveCharacterTextSplitter", FakeSplitter)

    splitter = LangChainDocumentSplitter(chunk_size=100, chunk_overlap=10)
    result = splitter.run(
        [Document(id="d1", content="source text", meta={"kind": "demo"}, score=0.5)]
    )

    assert calls == [{"chunk_size": 100, "chunk_overlap": 10}]
    assert [document.content for document in result["documents"]] == ["alpha", "beta"]
    assert [document.id for document in result["documents"]] == [
        "d1::chunk-0",
        "d1::chunk-1",
    ]
    assert [document.score for document in result["documents"]] == [0.5, 0.5]
    assert result["documents"][0].meta == {
        "kind": "demo",
        "source_document_id": "d1",
        "chunk_index": 0,
        "chunk_count": 2,
    }


def test_langchain_document_splitter_uses_optional_tokenizer_length(monkeypatch) -> None:
    calls = {"tokenizer_paths": [], "splitter_parameters": [], "lengths": []}

    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            return list(range(len(text.split())))

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str) -> FakeTokenizer:
            calls["tokenizer_paths"].append(path)
            return FakeTokenizer()

    class FakeSplitter:
        def __init__(self, *, chunk_size, chunk_overlap, length_function) -> None:
            calls["splitter_parameters"].append((chunk_size, chunk_overlap))
            calls["lengths"].append(length_function("one two three"))

        def split_text(self, text: str) -> list[str]:
            return [text]

    monkeypatch.setattr(splitter_module, "RecursiveCharacterTextSplitter", FakeSplitter)
    monkeypatch.setattr(splitter_module, "AutoTokenizer", FakeAutoTokenizer)

    splitter = LangChainDocumentSplitter(
        chunk_size=50,
        chunk_overlap=5,
        tokenizer_path="intfloat/e5-small-v2",
    )
    source = Document(id="d1", content="alpha beta")
    splitter.run([source])
    splitter.run([source])

    assert calls == {
        "tokenizer_paths": ["intfloat/e5-small-v2"],
        "splitter_parameters": [(50, 5)],
        "lengths": [3],
    }


def test_http_query_reformulator_posts_query_and_extracts_response(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": {"queries": ["expanded query", "alternate query"]}}

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(
        "retrieval_components.reformulation.http_query_reformulator.requests.post",
        fake_post,
    )

    reformulator = HttpQueryReformulator(
        url="https://example.test/reformulate",
        response_path="data.queries",
        extra_payload={"mode": "rewrite"},
        timeout=3.0,
    )
    source = Query(id="q1", content="original query", meta={"language": "en"})
    result = reformulator.run(source)

    assert result == {
        "query": source.with_content("expanded query"),
        "queries": [
            source.with_content("expanded query"),
            source.with_content("alternate query"),
        ],
    }
    assert calls == [
        {
            "url": "https://example.test/reformulate",
            "json": {"mode": "rewrite", "query": "original query"},
            "headers": {},
            "timeout": 3.0,
        }
    ]
