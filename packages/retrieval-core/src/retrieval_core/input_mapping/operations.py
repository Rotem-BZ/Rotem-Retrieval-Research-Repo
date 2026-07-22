"""Input-mapping resolution, generation, and persistence."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from haystack import Document
from omegaconf import DictConfig

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.utils.io import project_path, read_json, read_jsonl, write_json_atomic

INPUT_MAPPING_FILENAME = "input_mapping.json"
INPUT_MAPPING_METADATA_FILENAME = "meta.json"
RUN_ID_FORBIDDEN_CHARS = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}

GENERATION_KEYS = (
    "seed",
    "query_subset_size",
    "document_subset_size",
    "random_docs_per_query",
    "easy_negative_docs_per_query",
    "gold_passage_docs_per_query",
)


@dataclass(frozen=True)
class InferenceMapping:
    """Resolved query subset and candidates for one inference run."""

    queries: list[dict[str, Any]]
    candidate_ids_by_query: dict[str, list[str]]
    documents_by_id: dict[str, Document]
    default_candidate_ids: list[str] | None = None

    def candidate_ids(self, query_input: str) -> list[str]:
        if query_input in self.candidate_ids_by_query:
            return self.candidate_ids_by_query[query_input]
        if self.default_candidate_ids is not None:
            return self.default_candidate_ids
        raise KeyError(f"No candidate ids configured for query input {query_input!r}.")


@dataclass(frozen=True)
class GeneratedInputMapping:
    """Generated mapping payload plus sidecar metadata."""

    mapping: dict[str, list[str]]
    metadata: dict[str, Any]


def _document_from_record(record: dict[str, Any]) -> Document:
    EVALUATION_DATA_SCHEMA.validate_document(record)
    reserved_fields = {
        EVALUATION_DATA_SCHEMA.doc_id,
        "meta",
        "score",
        "embedding",
    }
    meta = {key: value for key, value in record.items() if key not in reserved_fields}
    meta.update(dict(record.get("meta") or {}))
    return Document(
        id=str(record[EVALUATION_DATA_SCHEMA.doc_id]),
        content=(
            str(record[EVALUATION_DATA_SCHEMA.text])
            if EVALUATION_DATA_SCHEMA.text in record
            else None
        ),
        meta=meta,
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def resolve_inference_mapping(cfg: DictConfig) -> InferenceMapping:
    """Resolve inference candidates from the dataset and an optional prepared mapping."""

    documents = read_jsonl(cfg.dataset.documents_path)
    queries = read_jsonl(cfg.dataset.queries_path)
    documents_by_id: dict[str, Document] = {}
    for record in documents:
        document = _document_from_record(record)
        document_id = str(document.id)
        if document_id in documents_by_id:
            raise ValueError(f"Duplicate document id in dataset: {document_id}")
        documents_by_id[document_id] = document
    queries_by_input: dict[str, dict[str, Any]] = {}
    for query in queries:
        EVALUATION_DATA_SCHEMA.validate_query(query)
        query_input = str(query[EVALUATION_DATA_SCHEMA.IN])
        if query_input in queries_by_input:
            raise ValueError(f"Duplicate query input in dataset: {query_input}")
        queries_by_input[query_input] = query
    all_document_ids = list(documents_by_id)

    mapping_path = configured_input_mapping_path(cfg)
    if mapping_path is None:
        return InferenceMapping(
            queries=queries,
            candidate_ids_by_query={},
            documents_by_id=documents_by_id,
            default_candidate_ids=all_document_ids,
        )

    return _resolve_file_mapping(
        mapping_path,
        queries_by_input=queries_by_input,
        documents_by_id=documents_by_id,
    )


def configured_input_mapping_path(cfg: DictConfig) -> Path | None:
    """Resolve the prepared mapping selected by folder name for inference."""

    selections = cfg.get("selections")
    configured_name = selections.get("input_mapping") if selections else None
    if configured_name is None or not str(configured_name).strip():
        return None
    if isinstance(configured_name, DictConfig):
        raise TypeError(
            "Inference selections.input_mapping must be a prepared mapping folder name, "
            "not an input-mapping recipe. Run prepare_mapping first."
        )
    normalized = validate_input_mapping_id(configured_name)
    return project_path(cfg.paths.input_mappings_dir) / normalized / INPUT_MAPPING_FILENAME


def validate_input_mapping_id(input_mapping_id: object) -> str:
    """Return an input-mapping id that is safe as one directory name."""

    normalized = str(input_mapping_id).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or Path(normalized).name != normalized
        or any(character in normalized for character in RUN_ID_FORBIDDEN_CHARS)
    ):
        raise ValueError(
            f"Input mapping id must be one folder name, got {input_mapping_id!r}."
        )
    return normalized


def discover_input_mapping_ids(
    input_mappings_dir: str | Path,
    *,
    dataset_name: str | None = None,
) -> list[str]:
    """Return completed prepared-mapping ids, optionally for one dataset."""

    root = project_path(input_mappings_dir)
    if not root.is_dir():
        return []

    input_mapping_ids: list[str] = []
    for directory in root.iterdir():
        mapping_path = directory / INPUT_MAPPING_FILENAME
        metadata_path = directory / INPUT_MAPPING_METADATA_FILENAME
        if not directory.is_dir() or not mapping_path.is_file() or not metadata_path.is_file():
            continue
        try:
            input_mapping_id = validate_input_mapping_id(directory.name)
            metadata = read_json(metadata_path)
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict):
            continue
        if dataset_name is not None and str(metadata.get("dataset", "")) != str(dataset_name):
            continue
        input_mapping_ids.append(input_mapping_id)
    return sorted(input_mapping_ids)


def _resolve_file_mapping(
    mapping_path: Path,
    *,
    queries_by_input: dict[str, dict[str, Any]],
    documents_by_id: dict[str, Document],
) -> InferenceMapping:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Prepared input mapping does not exist: {mapping_path}")
    raw_mapping = read_json(mapping_path)
    if not isinstance(raw_mapping, dict):
        raise ValueError("Input mapping JSON must be an object keyed by query input (`IN`).")
    candidate_ids_by_query: dict[str, list[str]] = {}
    for raw_query_input, raw_candidate_ids in raw_mapping.items():
        query_input = str(raw_query_input)
        if query_input not in queries_by_input:
            raise ValueError(f"Input mapping references unknown query input: {query_input}")
        if not isinstance(raw_candidate_ids, list):
            raise ValueError(f"Input mapping for query input {query_input} must be a list.")

        candidate_ids = [str(document_id) for document_id in raw_candidate_ids]
        missing_document_ids = [
            document_id for document_id in candidate_ids if document_id not in documents_by_id
        ]
        if missing_document_ids:
            raise ValueError(
                f"Input mapping for query input {query_input} references unknown document ids: "
                f"{missing_document_ids}"
            )
        candidate_ids_by_query[query_input] = list(dict.fromkeys(candidate_ids))
    selected_queries = [queries_by_input[query_input] for query_input in candidate_ids_by_query]

    return InferenceMapping(
        queries=selected_queries,
        candidate_ids_by_query=candidate_ids_by_query,
        documents_by_id=documents_by_id,
    )


def input_mapping_generation_params(mapping_cfg: DictConfig) -> dict[str, Any]:
    """Return the generation parameters represented by an input-mapping recipe."""

    params = {
        "seed": int(mapping_cfg.get("seed", 0)),
        "query_subset_size": (
            None
            if mapping_cfg.get("query_subset_size") is None
            else int(mapping_cfg.get("query_subset_size"))
        ),
        "document_subset_size": (
            None
            if mapping_cfg.get("document_subset_size") is None
            else int(mapping_cfg.get("document_subset_size"))
        ),
        "random_docs_per_query": int(mapping_cfg.get("random_docs_per_query", 0)),
        "easy_negative_docs_per_query": int(mapping_cfg.get("easy_negative_docs_per_query", 0)),
        "gold_passage_docs_per_query": int(mapping_cfg.get("gold_passage_docs_per_query", 0)),
    }
    return {key: params[key] for key in GENERATION_KEYS}


def prepared_mapping_dir(cfg: DictConfig) -> Path:
    """Return the run-id output directory for a prepared mapping."""

    run_id = cfg.stage.get("run_id")
    normalized = str(run_id).strip() if run_id is not None else ""
    if not normalized:
        raise ValueError("prepare_mapping requires a non-empty stage.run_id.")
    if (
        normalized in {".", ".."}
        or Path(normalized).name != normalized
        or any(character in normalized for character in RUN_ID_FORBIDDEN_CHARS)
    ):
        raise ValueError(f"stage.run_id must be one valid directory name, got {run_id!r}.")
    return project_path(cfg.paths.input_mappings_dir) / normalized


def prepared_mapping_path(cfg: DictConfig) -> Path:
    """Return the mapping JSON path for a prepare_mapping run."""

    return prepared_mapping_dir(cfg) / INPUT_MAPPING_FILENAME


def prepare_generated_input_mapping(cfg: DictConfig) -> tuple[GeneratedInputMapping, Path]:
    """Generate a run-id-scoped input mapping for later inference runs."""

    mapping_cfg = cfg.get("input_mapping_recipe")
    mapping_type = str(mapping_cfg.get("type", "")) if mapping_cfg else ""
    if mapping_type != "generated":
        raise ValueError(
            "prepare_mapping requires an input_mapping_recipe config with type: generated."
        )

    mapping_path = prepared_mapping_path(cfg)
    if mapping_path.parent.exists():
        raise FileExistsError(
            f"Input mapping run already exists; choose another stage.run_id: "
            f"{mapping_path.parent}"
        )

    generated = generate_input_mapping(
        dataset_name=str(cfg.dataset.name),
        documents=read_jsonl(cfg.dataset.documents_path),
        queries=read_jsonl(cfg.dataset.queries_path),
        qrels=read_jsonl(cfg.dataset.qrels_path),
        **input_mapping_generation_params(mapping_cfg),
    )
    generated.metadata["recipe_name"] = str(mapping_cfg.get("name", "generated"))
    generated.metadata["run_id"] = str(cfg.stage.run_id).strip()
    write_generated_mapping(generated, output_dir=mapping_path.parent, overwrite=False)
    return generated, mapping_path


def generate_input_mapping(
    *,
    dataset_name: str,
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    seed: int,
    query_subset_size: int | None = None,
    document_subset_size: int | None = None,
    random_docs_per_query: int = 0,
    easy_negative_docs_per_query: int = 0,
    gold_passage_docs_per_query: int = 0,
) -> GeneratedInputMapping:
    """Generate a stable candidate mapping from documents, queries, and qrels."""

    for name, value in {
        "random_docs_per_query": random_docs_per_query,
        "easy_negative_docs_per_query": easy_negative_docs_per_query,
        "gold_passage_docs_per_query": gold_passage_docs_per_query,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")
    rng = random.Random(seed)
    for document in documents:
        EVALUATION_DATA_SCHEMA.validate_document(document)
    for query in queries:
        EVALUATION_DATA_SCHEMA.validate_query(query)
    for qrel in qrels:
        EVALUATION_DATA_SCHEMA.validate_qrel(qrel)

    document_ids = [str(document[EVALUATION_DATA_SCHEMA.doc_id]) for document in documents]
    query_inputs = [str(query[EVALUATION_DATA_SCHEMA.IN]) for query in queries]
    if query_subset_size is None:
        selected_query_inputs = query_inputs
    else:
        if query_subset_size < 0:
            raise ValueError("queries subset size must be non-negative.")
        if query_subset_size > len(query_inputs):
            raise ValueError(
                f"Requested {query_subset_size} queries, but only {len(query_inputs)} exist."
            )
        selected_query_inputs = sorted(
            rng.sample(query_inputs, query_subset_size), key=query_inputs.index
        )

    annotated_by_query: dict[str, set[str]] = {}
    positive_by_query: dict[str, set[str]] = {}
    for qrel in qrels:
        query_input = str(qrel[EVALUATION_DATA_SCHEMA.IN])
        document_id = str(qrel[EVALUATION_DATA_SCHEMA.doc_id])
        annotated_by_query.setdefault(query_input, set()).add(document_id)
        if int(qrel[EVALUATION_DATA_SCHEMA.label]) > 0:
            positive_by_query.setdefault(query_input, set()).add(document_id)
    required_document_ids = set().union(
        *(annotated_by_query.get(query_input, set()) for query_input in selected_query_inputs)
    )
    if document_subset_size is None:
        active_document_ids = document_ids
    else:
        if document_subset_size < 0:
            raise ValueError("document_subset_size must be non-negative.")
        if document_subset_size < len(required_document_ids):
            raise ValueError(
                "document_subset_size is smaller than the number of annotated documents "
                "required for the selected queries."
            )
        if document_subset_size > len(document_ids):
            raise ValueError(
                f"Requested {document_subset_size} documents, but only {len(document_ids)} exist."
            )
        remaining_pool = [
            document_id for document_id in document_ids if document_id not in required_document_ids
        ]
        sampled = set(required_document_ids)
        sampled.update(rng.sample(remaining_pool, document_subset_size - len(sampled)))
        active_document_ids = [
            document_id for document_id in document_ids if document_id in sampled
        ]
    active_document_set = set(active_document_ids)

    annotated_anywhere = set().union(*annotated_by_query.values()) if annotated_by_query else set()
    easy_negative_pool = [
        document_id for document_id in active_document_ids if document_id not in annotated_anywhere
    ]
    if easy_negative_docs_per_query and not easy_negative_pool:
        raise ValueError("No easy negative documents exist for this dataset and document subset.")

    mapping: dict[str, list[str]] = {}
    for query_input in selected_query_inputs:
        included = [
            document_id
            for document_id in document_ids
            if document_id in annotated_by_query.get(query_input, set())
        ]
        missing_required = [
            document_id for document_id in included if document_id not in active_document_set
        ]
        if missing_required:
            raise ValueError(
                f"Document subset excludes annotated documents for query input {query_input}: "
                f"{missing_required}"
            )

        included_set = set(included)
        included.extend(
            _sample_new(
                active_document_ids,
                excluded=included_set,
                count=random_docs_per_query,
                rng=rng,
                label=f"random documents for query input {query_input}",
            )
        )
        included_set = set(included)
        included.extend(
            _sample_new(
                easy_negative_pool,
                excluded=included_set,
                count=easy_negative_docs_per_query,
                rng=rng,
                label=f"easy negative documents for query input {query_input}",
            )
        )
        included_set = set(included)
        current_annotations = annotated_by_query.get(query_input, set())
        positive_elsewhere = set().union(
            *(
                document_ids
                for other_query_input, document_ids in positive_by_query.items()
                if other_query_input != query_input
            )
        )
        gold_pool = [
            document_id
            for document_id in active_document_ids
            if document_id in positive_elsewhere and document_id not in current_annotations
        ]
        included.extend(
            _sample_new(
                gold_pool,
                excluded=included_set,
                count=gold_passage_docs_per_query,
                rng=rng,
                label=f"gold passage negative documents for query input {query_input}",
            )
        )
        mapping[query_input] = included

    candidate_counts = [len(candidate_ids) for candidate_ids in mapping.values()]
    metadata = {
        "dataset": dataset_name,
        "seed": seed,
        "query_subset_size": query_subset_size,
        "document_subset_size": document_subset_size,
        "random_docs_per_query": random_docs_per_query,
        "easy_negative_docs_per_query": easy_negative_docs_per_query,
        "gold_passage_docs_per_query": gold_passage_docs_per_query,
        "query_count": len(mapping),
        "active_document_count": len(active_document_ids),
        "candidate_count_min": min(candidate_counts, default=0),
        "candidate_count_max": max(candidate_counts, default=0),
        "candidate_count_mean": mean(candidate_counts) if candidate_counts else 0.0,
    }
    return GeneratedInputMapping(mapping=mapping, metadata=metadata)


def write_generated_mapping(
    generated: GeneratedInputMapping,
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a prepared mapping directory with fixed artifact names."""

    mapping_path = output_dir / INPUT_MAPPING_FILENAME
    metadata_path = metadata_path_for(mapping_path)
    if not overwrite:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing directory: {output_dir}")

    write_json_atomic(mapping_path, generated.mapping)
    write_json_atomic(metadata_path, generated.metadata)
    return mapping_path, metadata_path


def metadata_path_for(mapping_path: Path) -> Path:
    return mapping_path.with_name(INPUT_MAPPING_METADATA_FILENAME)


def _sample_new(
    pool: list[str],
    *,
    excluded: set[str],
    count: int,
    rng: random.Random,
    label: str,
) -> list[str]:
    if count == 0:
        return []
    available = [document_id for document_id in pool if document_id not in excluded]
    if len(available) < count:
        raise ValueError(f"Not enough {label}: requested {count}, available {len(available)}.")
    return rng.sample(available, count)
