import asyncio
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.stages.indexing import run_indexing
from retrieval_core.utils.io import write_jsonl


class CapturingPipeline:
    def __init__(self) -> None:
        self.data = None
        self.include_outputs_from = None
        self.concurrency_limit = None

    async def run_async(self, *, data, include_outputs_from, concurrency_limit):
        self.data = data
        self.include_outputs_from = include_outputs_from
        self.concurrency_limit = concurrency_limit
        return {"output": {}}


def _config(tmp_path: Path, documents_path: Path):
    index_path = tmp_path / "artifacts" / "indexes" / "test-index" / "index.jsonl"
    return OmegaConf.create(
        {
            "paths": {
                "indexes_dir": str(tmp_path / "artifacts" / "indexes"),
            },
            "dataset": {
                "name": "test-dataset",
                "documents_path": str(documents_path),
            },
            "selections": {"index_id": "test-index"},
            "pipeline": {
                "components": {
                    "indexer": {
                        "init_parameters": {
                            "output_path": str(index_path),
                        }
                    },
                    "output": {
                        "type": "retrieval_components.interfaces.stage_io.IndexingOutput"
                    },
                },
                "connections": [
                    {
                        "sender": "indexer.index_path",
                        "receiver": "output.index_path",
                    }
                ],
                "max_runs_per_component": 100,
                "metadata": {},
            },
            "runtime": {"concurrency_limit": 3},
            "stage": {
                "name": "indexing",
                "run_id": "test-run",
                "output_dir": str(tmp_path / "artifacts" / "runs" / "indexing" / "test-run"),
            },
        }
    )


def test_indexing_stage_loads_documents_and_supplies_fixed_pipeline_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {
                "doc_id": "d1",
                "text": "document text",
                "title": "Title",
                "meta": {"split": "test"},
                "score": 0.5,
                "embedding": [1.0, 0.0],
            }
        ],
    )
    pipeline = CapturingPipeline()
    monkeypatch.setattr(
        "retrieval_core.stages.indexing.load_async_pipeline",
        lambda _config: pipeline,
    )

    asyncio.run(run_indexing(_config(tmp_path, documents_path)))

    documents = pipeline.data["input"]["documents"]
    assert len(documents) == 1
    assert documents[0].id == "d1"
    assert documents[0].content is None
    assert documents[0].meta == {
        "text": "document text",
        "title": "Title",
        "split": "test",
    }
    assert documents[0].score == 0.5
    assert documents[0].embedding == [1.0, 0.0]
    assert pipeline.include_outputs_from == {"output"}
    assert pipeline.concurrency_limit == 3


def test_indexing_stage_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"doc_id": "duplicate", "text": "first"},
            {"doc_id": "duplicate", "text": "second"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate document id in dataset: duplicate"):
        asyncio.run(run_indexing(_config(tmp_path, documents_path)))


def test_indexing_stage_rejects_documents_without_ids(tmp_path: Path) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [{"text": "missing id"}],
    )

    with pytest.raises(ValueError, match="Document record is missing required fields"):
        asyncio.run(run_indexing(_config(tmp_path, documents_path)))
