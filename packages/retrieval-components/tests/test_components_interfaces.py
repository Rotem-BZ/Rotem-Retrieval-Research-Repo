from haystack import Document

from retrieval_components.interfaces import IndexingInput


def test_indexing_input_exposes_documents_without_copying_or_mutating() -> None:
    documents = [
        Document(
            id="d1",
            content=None,
            meta={"text": "document text"},
            score=0.5,
            embedding=[1.0, 0.0],
        )
    ]

    result = IndexingInput().run(documents)

    assert result["documents"] is documents
    assert result["documents"][0].meta == {"text": "document text"}
