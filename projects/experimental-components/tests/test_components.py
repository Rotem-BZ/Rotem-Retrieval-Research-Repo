from pathlib import Path

import pytest
import yaml
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie.recursive_splitter import (
    ChonkieRecursiveDocumentSplitter,
)
from haystack_integrations.components.preprocessors.chonkie.sentence_splitter import (
    ChonkieSentenceDocumentSplitter,
)
from haystack_integrations.components.preprocessors.chonkie.token_splitter import (
    ChonkieTokenDocumentSplitter,
)

from experimental_components import SourceDocumentIdAdapter
from retrieval_core.utils.pipelines import load_async_pipeline


PROJECT_DIR = Path(__file__).parents[1]


def test_source_document_id_adapter_preserves_document_fields() -> None:
    document = Document(
        id="chunk-1",
        content="evidence",
        meta={"source_id": "doc-1", "split_id": 0},
        score=0.75,
        embedding=[0.1, 0.2],
    )

    result = SourceDocumentIdAdapter().run([document])["documents"][0]

    assert result.id == "chunk-1"
    assert result.content == "evidence"
    assert result.score == 0.75
    assert result.embedding == [0.1, 0.2]
    assert result.meta == {
        "source_id": "doc-1",
        "source_document_id": "doc-1",
        "split_id": 0,
    }
    assert "source_document_id" not in document.meta


def test_source_document_id_adapter_rejects_missing_source_id() -> None:
    with pytest.raises(ValueError, match="missing meta.source_id"):
        SourceDocumentIdAdapter().run([Document(id="chunk-1", content="evidence")])


def test_source_document_id_adapter_rejects_conflicting_source_ids() -> None:
    document = Document(
        id="chunk-1",
        content="evidence",
        meta={"source_id": "doc-1", "source_document_id": "doc-2"},
    )

    with pytest.raises(ValueError, match="conflicting"):
        SourceDocumentIdAdapter().run([document])


@pytest.mark.parametrize(
    "splitter",
    [
        ChonkieTokenDocumentSplitter(
            tokenizer="character",
            chunk_size=40,
            chunk_overlap=5,
        ),
        ChonkieSentenceDocumentSplitter(
            tokenizer="character",
            chunk_size=40,
            chunk_overlap=5,
            min_characters_per_sentence=1,
        ),
        ChonkieRecursiveDocumentSplitter(
            tokenizer="character",
            chunk_size=40,
            min_characters_per_chunk=1,
        ),
    ],
    ids=["token", "sentence", "recursive"],
)
def test_non_model_chonkie_splitters_preserve_source_identity(splitter: object) -> None:
    source = Document(
        id="doc-1",
        content="First evidence sentence. Second evidence sentence. Third evidence sentence.",
    )

    chunks = splitter.run([source])["documents"]  # type: ignore[attr-defined]
    adapted = SourceDocumentIdAdapter().run(chunks)["documents"]

    assert adapted
    assert {chunk.meta["source_document_id"] for chunk in adapted} == {"doc-1"}


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/component/document_embedder/fastembed_sparse.yaml",
        "configs/component/query_embedder/fastembed_sparse.yaml",
    ],
)
def test_sparse_fastembed_component_configs_deserialize(relative_path: str) -> None:
    component_config = yaml.safe_load((PROJECT_DIR / relative_path).read_text(encoding="utf-8"))
    component_config["init_parameters"]["progress_bar"] = False

    pipeline = load_async_pipeline(
        {
            "components": {"embedder": component_config},
            "connections": [],
            "max_runs_per_component": 1,
        }
    )

    assert "embedder" in pipeline.graph.nodes
