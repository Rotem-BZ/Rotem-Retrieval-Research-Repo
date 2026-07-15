from haystack import Document

from retrieval_core.stages.inference import _build_query_inputs


def test_build_query_inputs_targets_fixed_inference_input_component() -> None:
    documents = [Document(id="d1", content="one")]

    assert _build_query_inputs(
        "hydra pipelines",
        candidate_document_ids=["d1"],
        candidate_documents=documents,
    ) == {
        "input": {
            "query": "hydra pipelines",
            "candidate_document_ids": ["d1"],
            "candidate_documents": documents,
        }
    }
