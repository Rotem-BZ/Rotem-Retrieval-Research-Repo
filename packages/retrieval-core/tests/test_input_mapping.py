from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from retrieval_core.input_mapping import (
    INPUT_MAPPING_FILENAME,
    INPUT_MAPPING_METADATA_FILENAME,
    discover_input_mapping_ids,
    generate_input_mapping,
    metadata_path_for,
    prepare_generated_input_mapping,
    prepared_mapping_path,
    resolve_inference_mapping,
    write_generated_mapping,
)
from retrieval_core.utils.io import read_json, write_json, write_jsonl


def _document(doc_id: str, text: str = "", **extra: Any) -> dict[str, Any]:
    return {"id": doc_id, "content": text, "meta": extra}


def _query(query_input: str, content: str = "", **extra: Any) -> dict[str, Any]:
    return {"id": f"query-{query_input}", "IN": query_input, "content": content, "meta": extra}


def _qrel(query_input: str, doc_id: str, label: int) -> dict[str, Any]:
    return {"IN": query_input, "document_id": doc_id, "label": label}


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

    assert [query.IN for query in mapping.queries] == ["q1", "q2"]
    assert mapping.candidate_ids_by_query == {}
    assert mapping.candidate_ids("q1") == ["d1", "d2", "d3", "d4", "d5"]
    assert mapping.candidate_ids("q2") == ["d1", "d2", "d3", "d4", "d5"]
    assert mapping.documents_by_id["d1"].meta["title"] == "Optional title"
    assert mapping.queries[0].meta["language"] == "en"


def test_multiple_queries_can_share_one_information_need(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, None)
    write_jsonl(
        cfg.dataset.queries_path,
        [
            {"id": "q1-a", "IN": "need-1", "content": "first wording", "meta": {}},
            {"id": "q1-b", "IN": "need-1", "content": "second wording", "meta": {}},
        ],
    )

    mapping = resolve_inference_mapping(cfg)

    assert [query.id for query in mapping.queries] == ["q1-a", "q1-b"]
    assert [query.IN for query in mapping.queries] == ["need-1", "need-1"]
    assert mapping.candidate_ids("need-1") == ["d1", "d2", "d3", "d4", "d5"]


def test_conflicting_duplicate_qrels_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting qrels"):
        generate_input_mapping(
            dataset_name="test",
            documents=[_document("d1")],
            queries=[_query("need-1")],
            qrels=[_qrel("need-1", "d1", 0), _qrel("need-1", "d1", 1)],
            seed=1,
        )


def test_inference_mapping_materializes_canonical_document_content(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, None)
    write_jsonl(
        cfg.dataset.documents_path,
        [
            {"id": "d1", "content": "document text", "meta": {"title": "metadata title"}}
        ],
    )
    write_jsonl(
        cfg.dataset.queries_path,
        [
            {"id": "external-q1", "IN": "q1", "content": "query text", "meta": {}}
        ],
    )

    mapping = resolve_inference_mapping(cfg)

    assert mapping.documents_by_id["d1"].content == "document text"
    assert mapping.documents_by_id["d1"].meta == {"title": "metadata title"}


def test_file_input_mapping_runs_only_mapped_queries(tmp_path: Path) -> None:
    mapping_path = (
        tmp_path / "artifacts" / "input_mappings" / "custom_mapping" / INPUT_MAPPING_FILENAME
    )
    write_json(mapping_path, {"q2": ["d3", "d4"]})
    cfg = _cfg(tmp_path, "custom_mapping")

    mapping = resolve_inference_mapping(cfg)

    assert [query.IN for query in mapping.queries] == ["q2"]
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
        _generated_recipe(
            seed=13,
            query_subset_size=1,
            easy_negative_docs_per_query=1,
            gold_passage_docs_per_query=1,
        ),
        run_id="toy_dev_small",
    )
    mapping_path = prepared_mapping_path(cfg)

    _generated, prepared_path = prepare_generated_input_mapping(cfg)
    inference_cfg = _cfg(tmp_path, "toy_dev_small")
    mapping = resolve_inference_mapping(inference_cfg)

    assert prepared_path == mapping_path
    assert mapping_path.name == INPUT_MAPPING_FILENAME
    assert mapping_path.parent.name == "toy_dev_small"
    assert mapping_path.exists()
    assert metadata_path_for(mapping_path).exists()
    assert metadata_path_for(mapping_path).name == INPUT_MAPPING_METADATA_FILENAME
    assert [query.IN for query in mapping.queries] == ["q2"]
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
        _generated_recipe(seed=13),
        run_id="existing",
    )
    prepare_generated_input_mapping(cfg)

    with pytest.raises(FileExistsError, match="choose another stage.run_id"):
        prepare_generated_input_mapping(cfg)


def test_prepare_mapping_requires_every_generation_parameter(tmp_path: Path) -> None:
    recipe = _generated_recipe()
    del recipe["include_annotated_docs"]
    cfg = _cfg(tmp_path, recipe, run_id="missing-parameter")

    with pytest.raises(ValueError, match="include_annotated_docs"):
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


def test_generated_mapping_can_skip_automatically_included_annotated_documents() -> None:
    generated = generate_input_mapping(
        dataset_name="toy",
        documents=DOCUMENTS,
        queries=QUERIES,
        qrels=QRELS,
        seed=1,
        include_annotated_docs=False,
    )

    assert generated.mapping == {"q1": [], "q2": []}
    assert generated.metadata["include_annotated_docs"] is False


def test_generated_mapping_uses_every_selected_document_for_each_selected_query() -> None:
    documents = [_document(f"d{index}") for index in range(1, 7)]
    queries = [_query(f"q{index}") for index in range(1, 4)]
    qrels = [_qrel(f"q{index}", f"d{index}", 1) for index in range(1, 4)]

    generated = generate_input_mapping(
        dataset_name="toy",
        documents=documents,
        queries=queries,
        qrels=qrels,
        seed=7,
        query_subset_size=2,
        document_subset_size=4,
        use_all_selected_documents_for_every_query=True,
        random_docs_per_query=None,
        easy_negative_docs_per_query=0,
        gold_passage_docs_per_query=None,
    )

    assert len(generated.mapping) == 2
    candidate_lists = list(generated.mapping.values())
    assert len(candidate_lists[0]) == 4
    assert candidate_lists[0] == candidate_lists[1]
    assert generated.metadata["query_count"] == 2
    assert generated.metadata["active_document_count"] == 4
    assert generated.metadata["candidate_count_min"] == 4
    assert generated.metadata["candidate_count_max"] == 4
    assert generated.metadata["use_all_selected_documents_for_every_query"] is True
    assert generated.metadata["random_docs_per_query"] is None


def test_document_subset_is_random_and_does_not_reserve_annotated_documents() -> None:
    documents = [_document(f"d{index}") for index in range(1, 6)]
    queries = [_query(f"q{index}") for index in range(1, 6)]
    qrels = [_qrel(f"q{index}", f"d{index}", 1) for index in range(1, 6)]

    generated = generate_input_mapping(
        dataset_name="toy",
        documents=documents,
        queries=queries,
        qrels=qrels,
        seed=7,
        document_subset_size=2,
    )

    selected_document_ids = {
        document_id
        for candidate_ids in generated.mapping.values()
        for document_id in candidate_ids
    }
    assert len(selected_document_ids) == 2
    assert generated.metadata["active_document_count"] == 2


@pytest.mark.parametrize(
    "sampling_field",
    [
        "random_docs_per_query",
        "easy_negative_docs_per_query",
        "gold_passage_docs_per_query",
    ],
)
def test_all_selected_documents_rejects_per_query_sampling(sampling_field: str) -> None:
    sampling_counts = {
        "random_docs_per_query": None,
        "easy_negative_docs_per_query": None,
        "gold_passage_docs_per_query": None,
        sampling_field: 1,
    }

    with pytest.raises(ValueError, match=sampling_field):
        generate_input_mapping(
            dataset_name="toy",
            documents=DOCUMENTS,
            queries=QUERIES,
            qrels=QRELS,
            seed=1,
            use_all_selected_documents_for_every_query=True,
            **sampling_counts,
        )


def test_prepared_recipe_accepts_null_sampling_counts_in_all_documents_mode(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        _generated_recipe(
            name="shared_pool",
            seed=13,
            query_subset_size=1,
            document_subset_size=3,
            use_all_selected_documents_for_every_query=True,
            random_docs_per_query=None,
            easy_negative_docs_per_query=None,
            gold_passage_docs_per_query=None,
        ),
        run_id="toy_shared_pool",
    )

    generated, _ = prepare_generated_input_mapping(cfg)

    assert len(generated.mapping) == 1
    assert len(next(iter(generated.mapping.values()))) == 3
    assert generated.metadata["use_all_selected_documents_for_every_query"] is True


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


def _generated_recipe(**overrides: Any) -> dict[str, Any]:
    recipe = {
        "type": "generated",
        "name": "dev_small",
        "seed": 0,
        "query_subset_size": None,
        "document_subset_size": None,
        "include_annotated_docs": True,
        "use_all_selected_documents_for_every_query": False,
        "random_docs_per_query": 0,
        "easy_negative_docs_per_query": 0,
        "gold_passage_docs_per_query": 0,
    }
    recipe.update(overrides)
    return recipe


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
