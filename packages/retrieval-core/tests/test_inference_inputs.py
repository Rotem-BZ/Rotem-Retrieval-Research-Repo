import asyncio
from typing import ClassVar

import pytest
from haystack import Document
from retrieval_components.dataclasses.query import Query

from retrieval_core.input_mapping import InferenceMapping
from retrieval_core.stages.inference import _run_queries, _run_query


def test_run_queries_concurrently_and_preserve_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingProgress:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.updates = 0
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

        def update(self) -> None:
            self.updates += 1

    class TrackingPipeline:
        def __init__(self) -> None:
            self.active_runs = 0
            self.max_active_runs = 0
            self.pipeline_limits: list[int] = []
            self.queries: list[Query] = []

        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            query = data["input"]["query"]
            self.queries.append(query)
            query_content = query.content
            assert query_content is not None
            self.active_runs += 1
            self.max_active_runs = max(self.max_active_runs, self.active_runs)
            self.pipeline_limits.append(concurrency_limit)
            await asyncio.sleep(
                {"first": 0.03, "second": 0.01, "third": 0}[query_content]
            )
            self.active_runs -= 1
            return {
                "output": {
                    "query": query,
                    "documents": [
                        Document(id=f"document-{query_content}", content=query_content)
                    ],
                }
            }

    pipeline = TrackingPipeline()
    monkeypatch.setattr("retrieval_core.stages.inference.tqdm", RecordingProgress)
    inference_mapping = InferenceMapping(
        queries=[
            Query(
                id="external-q1",
                IN="q1",
                content="first",
                meta={"language": "en", "filters": {"year": {"gte": 2020}}},
            ),
            Query(id="external-q2", IN="q2", content="second"),
            Query(id="external-q3", IN="q3", content="third"),
        ],
        candidate_ids_by_query={},
        documents_by_id={},
        default_candidate_ids=[],
    )

    predictions = asyncio.run(
        _run_queries(
            pipeline,  # type: ignore[arg-type]
            inference_mapping,
            query_concurrency_limit=2,
            pipeline_concurrency_limit=7,
        )
    )

    assert pipeline.max_active_runs == 2
    assert pipeline.pipeline_limits == [7, 7, 7]
    assert [query.content for query in pipeline.queries] == [
        "first",
        "second",
        "third",
    ]
    assert pipeline.queries[0].meta["language"] == "en"
    assert pipeline.queries[0].meta["filters"] == {"year": {"gte": 2020}}
    assert pipeline.queries[0].content == "first"
    assert [prediction["id"] for prediction in predictions] == [
        "external-q1",
        "external-q2",
        "external-q3",
    ]
    assert [prediction["results"][0]["document_id"] for prediction in predictions] == [
        "document-first",
        "document-second",
        "document-third",
    ]
    assert len(RecordingProgress.instances) == 1
    progress = RecordingProgress.instances[0]
    assert progress.kwargs == {
        "total": 3,
        "desc": "Inference",
        "unit": "query",
        "disable": False,
    }
    assert progress.updates == 3


def test_run_query_saves_only_reserved_extra_data() -> None:
    class Pipeline:
        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            query = data["input"]["query"]
            return {
                "output": {
                    "query": Query(
                        id=query.id,
                        IN=query.IN,
                        content="rewritten",
                        meta={"dataset": "discarded", "_extra_data": {"final_query": "rewritten"}},
                    ),
                    "documents": [
                        Document(
                            id="d1::chunk-2",
                            content="processed",
                            score=0.8,
                            meta={
                                "source_document_id": "d1",
                                "dataset": "discarded",
                                "_extra_data": {"chunk_index": 2},
                            },
                        )
                    ],
                }
            }

    query = Query(id="q1", IN="need-1", content="original")
    mapping = InferenceMapping(
        queries=[query], candidate_ids_by_query={}, documents_by_id={}, default_candidate_ids=[]
    )
    prediction = asyncio.run(
        _run_query(Pipeline(), mapping, query, pipeline_concurrency_limit=1)  # type: ignore[arg-type]
    )
    assert prediction == {
        "id": "q1",
        "data": {"final_query": "rewritten"},
        "results": [
            {
                "id": "d1::chunk-2",
                "document_id": "d1",
                "score": 0.8,
                "data": {"chunk_index": 2},
            }
        ],
    }


def test_run_queries_rejects_missing_canonical_query_content() -> None:
    class UnusedPipeline:
        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            raise AssertionError("Pipeline must not run for an invalid query record.")

    query = Query(id="external-q1", IN="q1")
    inference_mapping = InferenceMapping(
        queries=[query],
        candidate_ids_by_query={},
        documents_by_id={},
        default_candidate_ids=[],
    )

    with pytest.raises(ValueError, match="without content"):
        asyncio.run(
            _run_query(
                UnusedPipeline(),  # type: ignore[arg-type]
                inference_mapping,
                query,
                pipeline_concurrency_limit=1,
            )
        )
