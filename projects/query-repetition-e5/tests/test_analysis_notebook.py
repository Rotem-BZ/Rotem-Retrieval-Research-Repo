from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "query-repetition-e5-small-scifact"
    / "analysis.ipynb"
)


def load_notebook_namespace(path: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace: dict[str, object] = {"__file__": str(path), "__name__": "notebook_test"}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_notebook_joins_qrels_and_summarizes_queries(tmp_path: Path) -> None:
    namespace = load_notebook_namespace(NOTEBOOK_PATH)
    runs_dir = tmp_path / "artifacts" / "runs" / "inference"
    predictions_path = runs_dir / "run-1" / "predictions.json"
    qrels_path = tmp_path / "qrels.jsonl"

    write_json(
        predictions_path,
        {
            "q1": {
                "query": "first query",
                "documents": {
                    "d1::chunk-0": {
                        "content": "relevant chunk",
                        "score": 0.9,
                        "meta": {"source_document_id": "d1", "title": "Relevant"},
                    },
                    "d1::chunk-1": {
                        "content": "another relevant chunk",
                        "score": 0.8,
                        "meta": {"source_document_id": "d1"},
                    },
                    "d2": {"content": "unjudged", "score": 0.7, "meta": {}},
                },
            },
            "q2": {"query": "second query", "documents": {}},
        },
    )
    write_json(
        runs_dir / "run-1" / "manifest.json",
        {"artifacts": {"predictions": str(predictions_path)}},
    )
    qrels_path.write_text(
        '{"query_id":"q1","document_id":"d1","relevance":2}\n'
        '{"query_id":"q2","document_id":"d3","relevance":1}\n',
        encoding="utf-8",
    )

    predictions, query_summary, qrels = namespace["build_analysis_frames"](
        {"baseline": "run-1"}, qrels_path, runs_dir=runs_dir, project_root=tmp_path
    )

    assert list(predictions["rank"]) == [1, 2, 3]
    assert list(predictions["relevance"]) == [2, 2, 0]
    assert list(predictions["is_relevant"]) == [True, True, False]
    assert predictions.loc[0, "evaluation_document_id"] == "d1"

    first_query = query_summary.loc[query_summary["query_id"].eq("q1")].iloc[0]
    assert first_query["retrieved_document_count"] == 2
    assert first_query["retrieved_relevant_count"] == 1
    assert first_query["first_relevant_rank"] == 1
    assert first_query["reciprocal_rank"] == 1.0
    assert first_query["recall_at_run_depth"] == 1.0

    second_query = query_summary.loc[query_summary["query_id"].eq("q2")].iloc[0]
    assert second_query["retrieved_document_count"] == 0
    assert second_query["relevant_document_count"] == 1
    assert pd.isna(second_query["first_relevant_rank"])
    assert second_query["reciprocal_rank"] == 0.0
    assert second_query["recall_at_run_depth"] == 0.0
    assert len(qrels) == 2
