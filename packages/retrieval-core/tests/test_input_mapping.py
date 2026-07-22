from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.input_mapping import (
    INPUT_MAPPING_FILENAME,
    INPUT_MAPPING_METADATA_FILENAME,
    discover_input_mapping_ids,
    generate_input_mapping,
    metadata_path_for,
    prepared_mapping_path,
    prepare_generated_input_mapping,
    resolve_inference_mapping,
    write_generated_mapping,
)
from retrieval_core.utils.io import read_json, write_json, write_jsonl


def _document(doc_id: str, text: str = "", **extra: Any) -> dict[str, Any]:
    return {
        EVALUATION_DATA_SCHEMA.doc_id: doc_id,
        EVALUATION_DATA_SCHEMA.text: text,
        **extra,
    }


def _query(query_input: str, content: str = "", **extra: Any) -> dict[str, Any]:
    return {
        EVALUATION_DATA_SCHEMA.query_id: f"query-{query_input}",
        EVALUATION_DATA_SCHEMA.IN: query_input,
        EVALUATION_DATA_SCHEMA.query_content: content,
        **extra,
    }


def _qrel(query_input: str, doc_id: str, label: int) -> dict[str, Any]:
    return {
        EVALUATION_DATA_SCHEMA.IN: query_input,
        EVALUATION_DATA_SCHEMA.doc_id: doc_id,
        EVALUATION_DATA_SCHEMA.label: label,
    }


DOCUMENTS = [
    _document("d1", "positive for q1", title="Optional title"),
    _document("d2", "judged nonrelevant for q1"),
    _document("d3", "positive for q2"),
    _document("d4", "easy negative one"),
    _document("d5", "easy negative two"),
]
QUERIES = [
    _query("q1", "first query", language="en"),
    _query("q2", "second query"),
]
QRELS = [
    _qrel("q1", "d1", 1),
    _qrel("q1", "d2", 0),
    _qrel("q2", "d3", 1),
]


def test_full_input_mapping_runs_all_queries_against_all_documents(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, None)

    mapping = resolve_inference_mapping(cfg)

    assert [query[EVALUATION_DATA_SCHEMA.IN] for query in mapping.queries] == ["q1", "q2"]
    assert mapping.candidate_ids_by_query == {}
    assert mapping.candidate_ids("q1") == ["d1", "d2", "d3", "d4", "d5"]
    assert mapping.candidate_ids("q2") == ["d1", "d2", "d3", "d4", "d5"]
    assert mapping.documents_by_id["d1"].meta["title"] == "Optional title"
    assert mapping.queries[0]["language"] == "en"


def test_inference_mapping_accepts_metadata_only_content_records(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, None)
    write_jsonl(
        cfg.dataset.documents_path,
        [{EVALUATION_DATA_SCHEMA.doc_id: "d1", "body": {"text": "nested"}}],
    )
    write_jsonl(
        cfg.dataset.queries_path,
        [
            {
                EVALUATION_DATA_SCHEMA.query_id: "external-q1",
                EVALUATION_DATA_SCHEMA.IN: "q1",
                "question": "metadata query",
            }
        ],
    )

    mapping = resolve_inference_mapping(cfg)

    assert mapping.documents_by_id["d1"].content is None
    assert mapping.documents_by_id["d1"].meta == {"body": {"text": "nested"}}
    assert mapping.queries[0]["question"] == "metadata query"


def test_file_input_mapping_runs_only_mapped_queries(tmp_path: Path) -> None:
    mapping_path = (
        tmp_path / "artifacts" / "input_mappings" / "custom_mapping" / INPUT_MAPPING_FILENAME
    )
    write_json(mapping_path, {"q2": ["d3", "d4"]})
    cfg = _cfg(tmp_path, "custom_mapping")

    mapping = resolve_inference_mapping(cfg)

    assert [query[EVALUATION_DATA_SCHEMA.IN] for query in mapping.queries] == ["q2"]
    assert mapping.candidate_ids_by_query == {"q2": ["d3", "d4"]}
    mapped_document_ids = [
        mapping.documents_by_id[document_id].id for document_id in mapping.candidate_ids("q2")
    ]
    assert mapped_document_ids == [
        "d3",
        "d4",
    ]


def test_generated_recipe_is_prepared_in_run_id_directory(tmp_path: Path) -> None:
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
        run_id="toy_dev_small",
    )
    mapping_path = prepared_mapping_path(cfg)

    generated, prepared_path = prepare_generated_input_mapping(cfg)
    inference_cfg = _cfg(tmp_path, "toy_dev_small")
    mapping = resolve_inference_mapping(inference_cfg)

    assert prepared_path == mapping_path
    assert mapping_path.name == INPUT_MAPPING_FILENAME
    assert mapping_path.parent.name == "toy_dev_small"
    assert mapping_path.exists()
    assert metadata_path_for(mapping_path).exists()
    assert metadata_path_for(mapping_path).name == INPUT_MAPPING_METADATA_FILENAME
    assert [query[EVALUATION_DATA_SCHEMA.IN] for query in mapping.queries] == ["q2"]
    assert set(mapping.candidate_ids("q2")) >= {"d3"}
    metadata = read_json(metadata_path_for(mapping_path))
    assert metadata["dataset"] == "toy"
    assert metadata["recipe_name"] == "dev_small"
    assert "mapping_name" not in metadata
    assert metadata["run_id"] == "toy_dev_small"
    assert "recipe" not in metadata
    assert "recipe_hash" not in metadata
    assert "cache_key" not in metadata
    assert "sources" not in metadata


def test_inference_requires_existing_prepared_mapping(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "missing")

    with pytest.raises(FileNotFoundError, match="Prepared input mapping does not exist"):
        resolve_inference_mapping(cfg)


def test_discovers_completed_input_mappings_for_selected_dataset(tmp_path: Path) -> None:
    root = tmp_path / "input_mappings"
    for mapping_id, dataset_name in (
        ("toy-b", "toy"),
        ("toy-a", "toy"),
        ("other", "other"),
    ):
        output_dir = root / mapping_id
        write_json(output_dir / INPUT_MAPPING_FILENAME, {})
        write_json(output_dir / INPUT_MAPPING_METADATA_FILENAME, {"dataset": dataset_name})

    write_json(root / "incomplete" / INPUT_MAPPING_FILENAME, {})
    write_json(root / "invalid-metadata" / INPUT_MAPPING_FILENAME, {})
    (root / "invalid-metadata" / INPUT_MAPPING_METADATA_FILENAME).write_text(
        "not json",
        encoding="utf-8",
    )

    assert discover_input_mapping_ids(root, dataset_name="toy") == ["toy-a", "toy-b"]


@pytest.mark.parametrize("name", ["../mapping", "nested/mapping", "nested\\mapping"])
def test_inference_mapping_name_must_be_one_folder(name: str, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, name)

    with pytest.raises(ValueError, match="must be one folder name"):
        resolve_inference_mapping(cfg)


def test_prepare_mapping_requires_run_id(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, {"type": "generated", "name": "dev_small", "seed": 13})

    with pytest.raises(ValueError, match="requires a non-empty stage.run_id"):
        prepare_generated_input_mapping(cfg)


def test_prepare_mapping_refuses_to_overwrite_run_id(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        {"type": "generated", "name": "dev_small", "seed": 13},
        run_id="existing",
    )
    prepare_generated_input_mapping(cfg)

    with pytest.raises(FileExistsError, match="choose another stage.run_id"):
        prepare_generated_input_mapping(cfg)


def test_generated_mapping_includes_judged_random_easy_and_gold_passage_negatives() -> None:
    documents = [
        _document("d1", "positive for q1"),
        _document("d2", "judged nonrelevant for q1"),
        _document("d3", "positive for q2"),
        _document("d4", "positive for q3"),
        _document("d5", "easy negative one"),
        _document("d6", "easy negative two"),
        _document("d7", "easy negative three"),
    ]
    queries = [
        _query("q1", "first query"),
        _query("q2", "second query"),
        _query("q3", "third query"),
    ]
    qrels = [
        _qrel("q1", "d1", 1),
        _qrel("q1", "d2", 0),
        _qrel("q2", "d3", 1),
        _qrel("q3", "d4", 1),
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
            _qrel("q1", "d1", 1),
            _qrel("q2", "d1", 1),
            _qrel("q2", "d3", 1),
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
            documents=[_document("d1"), _document("d2")],
            queries=[_query("q1")],
            qrels=[
                _qrel("q1", "d1", 1),
                _qrel("q2", "d2", 1),
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
    output_dir = tmp_path / "dev"

    written_mapping, written_metadata = write_generated_mapping(
        generated,
        output_dir=output_dir,
    )

    assert written_mapping == output_dir / INPUT_MAPPING_FILENAME
    assert written_metadata == output_dir / INPUT_MAPPING_METADATA_FILENAME
    assert written_mapping.exists()
    assert written_metadata.exists()


def _cfg(tmp_path: Path, input_mapping: object, *, run_id: str | None = None):
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
            "selections": {
                "input_mapping": None if isinstance(input_mapping, dict) else input_mapping
            },
            "input_mapping_recipe": input_mapping if isinstance(input_mapping, dict) else None,
            "paths": {"input_mappings_dir": str(tmp_path / "artifacts" / "input_mappings")},
            "stage": {"run_id": run_id},
        }
    )
