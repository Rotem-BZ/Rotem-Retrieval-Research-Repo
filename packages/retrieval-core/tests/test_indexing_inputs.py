import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.stages.indexing import IndexingStage, IndexWriter
from retrieval_core.utils.hashing import file_sha256
from retrieval_core.utils.io import write_jsonl


class CapturingPipeline:
    def __init__(self, *, fail_on_batch: int | None = None) -> None:
        self.writer = CapturingWriter()
        self.batches = []
        self.include_outputs_from = None
        self.concurrency_limit = None
        self.fail_on_batch = fail_on_batch

    async def run_async(self, *, data, include_outputs_from, concurrency_limit):
        documents = list(data["input"]["documents"])
        self.batches.append(documents)
        self.include_outputs_from = include_outputs_from
        self.concurrency_limit = concurrency_limit
        if self.fail_on_batch == len(self.batches):
            raise RuntimeError("batch failed")
        return {
            "output": {
                "index_path": self.writer.output_path,
                "indexed_count": len(documents),
            }
        }

    def get_component(self, name: str):
        assert name == "indexer"
        return self.writer


class CapturingWriter:
    def __init__(self) -> None:
        self.output_path = ""

    @asynccontextmanager
    async def write_session(self):
        yield
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def _capture_pipeline(pipeline: CapturingPipeline):
    def load_pipeline(config):
        pipeline.writer.output_path = str(
            config["components"]["indexer"]["init_parameters"]["output_path"]
        )
        return pipeline

    return load_pipeline


def test_capturing_writer_implements_index_writer() -> None:
    assert isinstance(CapturingWriter(), IndexWriter)


def _config(tmp_path: Path, documents_path: Path):
    index_path = tmp_path / "artifacts" / "indexes" / "test-index" / "index.json"
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
                        "type": (
                            "retrieval_components.indexing."
                            "persisted_in_memory_document_indexer."
                            "PersistedInMemoryDocumentIndexer"
                        ),
                        "init_parameters": {
                            "output_path": str(index_path),
                        },
                    },
                    "output": {"type": "retrieval_components.interfaces.indexing.IndexingOutput"},
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
            "runtime": {
                "concurrency_limit": 3,
                "indexing_batch_size": 2,
            },
            "stage": {
                "name": "indexing",
                "run_id": "test-run",
                "output_dir": str(tmp_path / "artifacts" / "runs" / "indexing" / "test-run"),
            },
        }
    )


def _run_stage(cfg):
    stage = IndexingStage(cfg)
    stage.prepare()
    return asyncio.run(stage.run())


def test_indexing_stage_loads_documents_and_supplies_batched_pipeline_input(
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
        _capture_pipeline(pipeline),
    )

    result = _run_stage(_config(tmp_path, documents_path))

    documents = pipeline.batches[0]
    assert len(documents) == 1
    assert documents[0].id == "d1"
    assert documents[0].content == "document text"
    assert documents[0].meta == {
        "title": "Title",
        "split": "test",
    }
    assert documents[0].score == 0.5
    assert documents[0].embedding == [1.0, 0.0]
    assert pipeline.include_outputs_from == {"output"}
    assert pipeline.concurrency_limit == 3
    assert result["source_document_count"] == 1
    assert result["batch_count"] == 1
    assert result["output"]["indexed_count"] == 1
    assert Path(result["output"]["index_path"]).is_file()


def test_indexing_stage_streams_multiple_batches_and_hashes_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"doc_id": "d1", "text": "one"},
            {"doc_id": "d2", "text": "two"},
            {"doc_id": "d3", "text": "three"},
            {"doc_id": "d4", "text": "four"},
            {"doc_id": "d5", "text": "five"},
        ],
    )
    pipeline = CapturingPipeline()
    monkeypatch.setattr(
        "retrieval_core.stages.indexing.load_async_pipeline",
        _capture_pipeline(pipeline),
    )
    cfg = _config(tmp_path, documents_path)

    result = _run_stage(cfg)

    assert [[document.id for document in batch] for batch in pipeline.batches] == [
        ["d1", "d2"],
        ["d3", "d4"],
        ["d5"],
    ]
    assert result["source_document_count"] == 5
    assert result["batch_count"] == 3
    assert result["output"]["indexed_count"] == 5
    manifest = json.loads(
        (Path(cfg.stage.output_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["inputs"]["documents_sha256"] == file_sha256(documents_path)
    assert manifest["inputs"]["batch_count"] == 3


def test_indexing_stage_removes_temporary_index_when_a_later_batch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"doc_id": "d1", "text": "one"},
            {"doc_id": "d2", "text": "two"},
            {"doc_id": "d3", "text": "three"},
        ],
    )
    pipeline = CapturingPipeline(fail_on_batch=2)
    monkeypatch.setattr(
        "retrieval_core.stages.indexing.load_async_pipeline",
        _capture_pipeline(pipeline),
    )
    cfg = _config(tmp_path, documents_path)

    with pytest.raises(RuntimeError, match="batch failed"):
        _run_stage(cfg)

    canonical_path = Path(cfg.pipeline.components.indexer.init_parameters.output_path)
    assert not canonical_path.exists()
    assert not list(canonical_path.parent.glob(f".{canonical_path.name}.*.tmp"))


def test_indexing_stage_writes_an_empty_index_for_an_empty_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents_path = write_jsonl(tmp_path / "documents.jsonl", [])
    pipeline = CapturingPipeline()
    monkeypatch.setattr(
        "retrieval_core.stages.indexing.load_async_pipeline",
        _capture_pipeline(pipeline),
    )

    result = _run_stage(_config(tmp_path, documents_path))

    assert pipeline.batches == []
    assert result["source_document_count"] == 0
    assert result["batch_count"] == 0
    assert result["output"]["indexed_count"] == 0
    assert Path(result["output"]["index_path"]).read_text(encoding="utf-8") == "{}"


def test_indexing_stage_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [
            {"doc_id": "duplicate", "text": "first"},
            {"doc_id": "duplicate", "text": "second"},
        ],
    )

    with pytest.raises(ValueError, match="Duplicate document id in dataset: duplicate"):
        _run_stage(_config(tmp_path, documents_path))


def test_indexing_stage_rejects_documents_without_ids(tmp_path: Path) -> None:
    documents_path = write_jsonl(
        tmp_path / "documents.jsonl",
        [{"text": "missing id"}],
    )

    with pytest.raises(ValueError, match="Document record is missing required fields"):
        _run_stage(_config(tmp_path, documents_path))
