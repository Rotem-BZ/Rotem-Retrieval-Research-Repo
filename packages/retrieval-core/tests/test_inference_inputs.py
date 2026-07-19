import asyncio

from haystack import Document

from retrieval_core.input_mapping import InferenceMapping
from retrieval_core.stages.inference import _run_queries


def test_run_queries_concurrently_and_preserve_input_order() -> None:
    class TrackingPipeline:
        def __init__(self) -> None:
            self.active_runs = 0
            self.max_active_runs = 0
            self.pipeline_limits: list[int] = []

        async def run_async(self, *, data, include_outputs_from, concurrency_limit):
            query = data["input"]["query"]
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
            {"id": "q1", "text": "first"},
            {"id": "q2", "text": "second"},
            {"id": "q3", "text": "third"},
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
    assert [prediction["query_id"] for prediction in predictions] == ["q1", "q2", "q3"]
