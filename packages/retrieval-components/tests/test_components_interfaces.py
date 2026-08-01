from haystack import Document

from retrieval_components.interfaces.indexing import IndexingInput
from retrieval_components.interfaces.inference import InferenceInput, InferenceOutput


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


def test_stage_interfaces_are_split_into_stage_modules() -> None:
    assert IndexingInput.__module__ == "retrieval_components.interfaces.indexing"
    assert InferenceInput.__module__ == "retrieval_components.interfaces.inference"
    assert InferenceOutput.__module__ == "retrieval_components.interfaces.inference"
