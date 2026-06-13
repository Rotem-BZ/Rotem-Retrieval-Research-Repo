import sys
from types import ModuleType

from haystack import Document

from retrieval_research.components.chunking import LangChainDocumentSplitter
from retrieval_research.components.reformulation import HttpQueryReformulator


def test_langchain_document_splitter_uses_configured_splitter(monkeypatch) -> None:
    fake_module = ModuleType("langchain_text_splitters")

    class FakeSplitter:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def split_text(self, text: str) -> list[str]:
            return [part.strip() for part in text.split(self.marker)]

    fake_module.FakeSplitter = FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_module)

    splitter = LangChainDocumentSplitter(
        splitter_type="FakeSplitter",
        splitter_kwargs={"marker": "|"},
    )
    result = splitter.run([Document(id="d1", content="alpha | beta", meta={"kind": "demo"})])

    assert [document.content for document in result["documents"]] == ["alpha", "beta"]
    assert result["documents"][0].meta == {
        "kind": "demo",
        "source_document_id": "d1",
        "chunk_index": 0,
        "chunk_count": 2,
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

    monkeypatch.setattr("retrieval_research.components.reformulation.http.requests.post", fake_post)

    reformulator = HttpQueryReformulator(
        url="https://example.test/reformulate",
        response_path="data.queries",
        extra_payload={"mode": "rewrite"},
        timeout=3.0,
    )
    result = reformulator.run("original query")

    assert result == {
        "query": "expanded query",
        "queries": ["expanded query", "alternate query"],
    }
    assert calls == [
        {
            "url": "https://example.test/reformulate",
            "json": {"mode": "rewrite", "query": "original query"},
            "headers": {},
            "timeout": 3.0,
        }
    ]
