import asyncio

from haystack import Document

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.input_mapping import InferenceMapping
from retrieval_core.stages.inference import _run_queries


def test_run_queries_concurrently_and_preserve_input_order() -> None:
    class TrackingPipeline:
        def __init__(self) -> None:
            self.active_runs = 0
            self.max_active_runs = 0
            self.pipeline_limits: list[int] = []
            self.query_metas: list[dict] = []

        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            query = data["input"]["query"]
            self.query_metas.append(data["input"]["query_meta"])
            self.active_runs += 1
            self.max_active_runs = max(self.max_active_runs, self.active_runs)
            self.pipeline_limits.append(concurrency_limit)
            await asyncio.sleep({"first": 0.03, "second": 0.01, "third": 0}[query])
            self.active_runs -= 1
            return {
                "output": {
                    "documents": [Document(id=f"document-{query}", content=query)],
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
    assert [meta[EVALUATION_DATA_SCHEMA.query_content] for meta in pipeline.query_metas] == [
        "first",
        "second",
        "third",
    ]
    assert pipeline.query_metas[0]["language"] == "en"
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


def test_run_queries_accepts_parser_rendered_content_when_raw_content_is_missing() -> None:
    class MetadataQueryPipeline:
        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            assert data["input"]["query"] == ""
            return {
                "output": {
                    "query_content": data["input"]["query_meta"]["question"],
                    "documents": [],
                }
            }

    inference_mapping = InferenceMapping(
        queries=[
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q1",
                EVALUATION_DATA_SCHEMA.IN: "q1",
                "question": "render me",
            }
        ],
        candidate_ids_by_query={},
        documents_by_id={},
        default_candidate_ids=[],
    )

    predictions = asyncio.run(
        _run_queries(
            MetadataQueryPipeline(),  # type: ignore[arg-type]
            inference_mapping,
            query_concurrency_limit=1,
            pipeline_concurrency_limit=1,
        )
    )

    assert predictions[0][EVALUATION_DATA_SCHEMA.query_content] == "render me"
