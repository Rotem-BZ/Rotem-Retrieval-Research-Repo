import asyncio

import pytest
from haystack import Document

from retrieval_components.dataclasses.query import Query
from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.input_mapping import InferenceMapping
from retrieval_core.stages.inference import _run_queries, _run_query


def test_run_queries_concurrently_and_preserve_input_order() -> None:
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
    inference_mapping = InferenceMapping(
        queries=[
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q1",
                EVALUATION_DATA_SCHEMA.IN: "q1",
                EVALUATION_DATA_SCHEMA.query_content: "first",
                "language": "en",
                "filters": {"year": {"gte": 2020}},
            },
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q2",
                EVALUATION_DATA_SCHEMA.IN: "q2",
                EVALUATION_DATA_SCHEMA.query_content: "second",
            },
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q3",
                EVALUATION_DATA_SCHEMA.IN: "q3",
                EVALUATION_DATA_SCHEMA.query_content: "third",
            },
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
    assert EVALUATION_DATA_SCHEMA.query_content not in pipeline.queries[0].meta
    assert pipeline.queries[0].meta["language"] == "en"
    assert pipeline.queries[0].meta["filters"] == {"year": {"gte": 2020}}
    assert pipeline.queries[0].content == "first"
    assert [prediction[EVALUATION_DATA_SCHEMA.query_id] for prediction in predictions] == [
        "external-q1",
        "external-q2",
        "external-q3",
    ]
    assert [prediction[EVALUATION_DATA_SCHEMA.IN] for prediction in predictions] == [
        "q1",
        "q2",
        "q3",
    ]


def test_run_queries_rejects_missing_canonical_query_content() -> None:
    class UnusedPipeline:
        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            raise AssertionError("Pipeline must not run for an invalid query record.")

    query = {
        EVALUATION_DATA_SCHEMA.query_id: "external-q1",
        EVALUATION_DATA_SCHEMA.IN: "q1",
    }
    inference_mapping = InferenceMapping(
        queries=[query],
        candidate_ids_by_query={},
        documents_by_id={},
        default_candidate_ids=[],
    )

    with pytest.raises(ValueError, match="query_content"):
        asyncio.run(
            _run_query(
                UnusedPipeline(),  # type: ignore[arg-type]
                inference_mapping,
                query,
                pipeline_concurrency_limit=1,
            )
        )
