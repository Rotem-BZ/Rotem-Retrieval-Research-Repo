from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.input_mapping import (
    generate_input_mapping,
    input_mapping_cache_key,
    input_mapping_recipe_hash,
    materialized_mapping_path,
    metadata_path_for,
    prepare_generated_input_mapping,
    resolve_inference_mapping,
    write_generated_mapping,
)
from retrieval_core.io import read_json, write_json, write_jsonl


DOCUMENTS = [
    {"id": "d1", "content": "positive for q1"},
    {"id": "d2", "content": "judged nonrelevant for q1"},
    {"id": "d3", "content": "positive for q2"},
    {"id": "d4", "content": "easy negative one"},
    {"id": "d5", "content": "easy negative two"},
]
QUERIES = [
    {"id": "q1", "text": "first query"},
    {"id": "q2", "text": "second query"},
]
QRELS = [
    {"query_id": "q1", "document_id": "d1", "relevance": 1},
    {"query_id": "q1", "document_id": "d2", "relevance": 0},
    {"query_id": "q2", "document_id": "d3", "relevance": 1},
]


def test_full_input_mapping_runs_all_queries_against_all_documents(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, {"type": "full_dataset"})

    mapping = resolve_inference_mapping(cfg)

    assert [query["id"] for query in mapping.queries] == ["q1", "q2"]
    assert mapping.candidate_ids_by_query == {}
    assert mapping.candidate_ids("q1") == ["d1", "d2", "d3", "d4", "d5"]
    assert mapping.candidate_ids("q2") == ["d1", "d2", "d3", "d4", "d5"]


def test_file_input_mapping_runs_only_mapped_queries(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    write_json(mapping_path, {"q2": ["d3", "d4"]})
    cfg = _cfg(
        tmp_path,
        {
            "type": "file",
            "dataset": "toy",
            "path": str(mapping_path),
        },
    )

    mapping = resolve_inference_mapping(cfg)

    assert [query["id"] for query in mapping.queries] == ["q2"]
    assert mapping.candidate_ids_by_query == {"q2": ["d3", "d4"]}
    assert [document.id for document in mapping.candidate_documents("q2")] == ["d3", "d4"]


def test_generated_recipe_is_prepared_explicitly_and_reused(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        {
            "type": "generated",
            "name": "dev_small",
            "seed": 13,
            "query_subset_size": 1,
            "random_docs_per_query": 0,
            "easy_negative_docs_per_query": 1,
            "gold_passage_docs_per_query": 1,
        },
    )
    mapping_path = materialized_mapping_path(cfg, cfg.input_mapping)

    generated, prepared_path, reused = prepare_generated_input_mapping(cfg)
    mapping = resolve_inference_mapping(cfg)
    reused_generated, reused_path, reused_again = prepare_generated_input_mapping(cfg)

    assert prepared_path == mapping_path
    assert reused is False
    assert reused_path == mapping_path
    assert reused_again is True
    assert reused_generated.mapping == generated.mapping
    assert mapping_path.exists()
    assert metadata_path_for(mapping_path).exists()
    assert [query["id"] for query in mapping.queries] == ["q2"]
    assert set(mapping.candidate_ids("q2")) >= {"d3"}
    metadata = read_json(metadata_path_for(mapping_path))
    assert metadata["dataset"] == "toy"
    assert metadata["mapping_name"] == "dev_small"
    assert metadata["recipe_hash"] == input_mapping_recipe_hash(cfg.input_mapping)
    assert metadata["cache_key"] == input_mapping_cache_key(cfg, cfg.input_mapping)


def test_generated_recipe_requires_explicit_preparation(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        {
            "type": "generated",
            "name": "dev_small",
            "seed": 13,
        },
    )

    with pytest.raises(FileNotFoundError, match="stage prepare_mapping"):
        resolve_inference_mapping(cfg)


def test_generated_mapping_includes_judged_random_easy_and_gold_passage_negatives() -> None:
    documents = [
        {"id": "d1", "content": "positive for q1"},
        {"id": "d2", "content": "judged nonrelevant for q1"},
        {"id": "d3", "content": "positive for q2"},
        {"id": "d4", "content": "positive for q3"},
        {"id": "d5", "content": "easy negative one"},
        {"id": "d6", "content": "easy negative two"},
        {"id": "d7", "content": "easy negative three"},
    ]
    queries = [
        {"id": "q1", "text": "first query"},
        {"id": "q2", "text": "second query"},
        {"id": "q3", "text": "third query"},
    ]
    qrels = [
        {"query_id": "q1", "document_id": "d1", "relevance": 1},
        {"query_id": "q1", "document_id": "d2", "relevance": 0},
        {"query_id": "q2", "document_id": "d3", "relevance": 1},
        {"query_id": "q3", "document_id": "d4", "relevance": 1},
    ]

    generated = generate_input_mapping(
        dataset_name="toy",
        documents=documents,
        queries=queries,
        qrels=qrels,
        seed=7,
        random_docs_per_query=1,
        easy_negative_docs_per_query=1,
        gold_passage_docs_per_query=1,
    )

    candidates = generated.mapping["q1"]
    assert {"d1", "d2"}.issubset(candidates)
    assert len(candidates) == 5
    assert set(candidates) & {"d3", "d4"}
    assert set(candidates) & {"d5", "d6", "d7"}
    assert generated.metadata["query_count"] == 3
    assert generated.metadata["candidate_count_min"] == 4
    assert generated.metadata["candidate_count_max"] == 5


def test_gold_passage_negatives_exclude_documents_annotated_for_current_query() -> None:
    generated = generate_input_mapping(
        dataset_name="toy",
        documents=DOCUMENTS,
        queries=QUERIES,
        qrels=[
            {"query_id": "q1", "document_id": "d1", "relevance": 1},
            {"query_id": "q2", "document_id": "d1", "relevance": 1},
            {"query_id": "q2", "document_id": "d3", "relevance": 1},
        ],
        seed=1,
        query_subset_size=1,
        gold_passage_docs_per_query=1,
    )

    assert generated.mapping["q1"] == ["d1", "d3"]


def test_easy_negatives_raise_when_no_unannotated_documents_exist() -> None:
    with pytest.raises(ValueError, match="No easy negative documents"):
        generate_input_mapping(
            dataset_name="tiny",
            documents=[{"id": "d1"}, {"id": "d2"}],
            queries=[{"id": "q1"}],
            qrels=[
                {"query_id": "q1", "document_id": "d1", "relevance": 1},
                {"query_id": "q2", "document_id": "d2", "relevance": 1},
            ],
            seed=1,
            easy_negative_docs_per_query=1,
        )


def test_mapping_metadata_is_written_as_sidecar(tmp_path: Path) -> None:
    generated = generate_input_mapping(
        dataset_name="toy",
        documents=DOCUMENTS,
        queries=QUERIES,
        qrels=QRELS,
        seed=1,
    )
    mapping_path = tmp_path / "dev.json"

    written_mapping, written_metadata = write_generated_mapping(
        generated,
        mapping_path=mapping_path,
    )

    assert written_mapping == mapping_path
    assert written_metadata == metadata_path_for(mapping_path)
    assert written_mapping.exists()
    assert written_metadata.exists()


def _cfg(tmp_path: Path, input_mapping: dict):
    documents_path = tmp_path / "documents.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    write_jsonl(documents_path, DOCUMENTS)
    write_jsonl(queries_path, QUERIES)
    write_jsonl(qrels_path, QRELS)

    return OmegaConf.create(
        {
            "dataset": {
                "name": "toy",
                "documents_path": str(documents_path),
                "queries_path": str(queries_path),
                "qrels_path": str(qrels_path),
            },
            "input_mapping": input_mapping,
            "paths": {"input_mappings_dir": str(tmp_path / "artifacts" / "input_mappings")},
        }
    )
