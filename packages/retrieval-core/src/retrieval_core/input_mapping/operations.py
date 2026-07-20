"""Input-mapping resolution, generation, caching, and persistence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from haystack import Document
from omegaconf import DictConfig

from retrieval_core.data_schema import EVALUATION_DATA_SCHEMA
from retrieval_core.utils.hashing import file_sha256, sha256_text
from retrieval_core.utils.io import project_path, read_json, read_jsonl, write_json_atomic

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
        EVALUATION_DATA_SCHEMA.text,
        "meta",
        "score",
        "embedding",
    }
    meta = {key: value for key, value in record.items() if key not in reserved_fields}
    meta.update(dict(record.get("meta") or {}))
    return Document(
        id=str(record[EVALUATION_DATA_SCHEMA.doc_id]),
        content=str(record[EVALUATION_DATA_SCHEMA.text]),
        meta=meta,
        score=record.get("score"),
        embedding=record.get("embedding"),
    )


def resolve_inference_mapping(cfg: DictConfig) -> InferenceMapping:
    """Resolve configured inference candidates from dataset files and input_mapping config."""

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

    mapping_cfg = cfg.get("input_mapping")
    mapping_type = str(mapping_cfg.get("type", "full_dataset")) if mapping_cfg else "full_dataset"

    if mapping_type == "full_dataset":
        return InferenceMapping(
            queries=queries,
            candidate_ids_by_query={},
            documents_by_id=documents_by_id,
            default_candidate_ids=all_document_ids,
        )

    if mapping_type == "generated":
        mapping_path = materialized_mapping_path(cfg, mapping_cfg)
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Generated input mapping is not prepared: {mapping_path}. Run: "
                f"stage prepare_mapping dataset={cfg.dataset.name} "
                f"input_mapping={mapping_cfg.get('name', 'generated')}"
            )
        return _resolve_file_mapping(
            mapping_path,
            mapping_cfg=mapping_cfg,
            cfg=cfg,
            queries_by_input=queries_by_input,
            documents_by_id=documents_by_id,
        )

    if mapping_type != "file":
        raise ValueError(f"Unsupported input mapping type: {mapping_type!r}")

    return _resolve_file_mapping(
        project_path(mapping_cfg.path),
        mapping_cfg=mapping_cfg,
        cfg=cfg,
        queries_by_input=queries_by_input,
        documents_by_id=documents_by_id,
    )


def validate_input_mapping_config(cfg: DictConfig) -> Path | None:
    """Return the configured input-mapping path, if it has one."""

    mapping_cfg = cfg.get("input_mapping")
    mapping_type = str(mapping_cfg.get("type", "full_dataset")) if mapping_cfg else "full_dataset"
    if mapping_type == "full_dataset":
        return None
    if mapping_type == "file":
        path = project_path(mapping_cfg.path)
    elif mapping_type == "generated":
        path = materialized_mapping_path(cfg, mapping_cfg)
    else:
        raise ValueError(f"Unsupported input mapping type: {mapping_type!r}")
    return path


def _resolve_file_mapping(
    mapping_path: Path,
    *,
    mapping_cfg: DictConfig,
    cfg: DictConfig,
    queries_by_input: dict[str, dict[str, Any]],
    documents_by_id: dict[str, Document],
) -> InferenceMapping:
    configured_dataset = mapping_cfg.get("dataset")
    if configured_dataset is not None and str(configured_dataset) != str(cfg.dataset.name):
        raise ValueError(
            "Input mapping dataset does not match selected dataset: "
            f"{configured_dataset!r} != {cfg.dataset.name!r}."
        )

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


def input_mapping_recipe_hash(mapping_cfg: DictConfig) -> str:
    """Return a stable short hash for generation-affecting recipe parameters."""

    payload = json.dumps(
        input_mapping_generation_params(mapping_cfg),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)[:12]


def input_mapping_cache_key(
    cfg: DictConfig,
    mapping_cfg: DictConfig,
    *,
    source_fingerprints: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Hash the generation recipe together with the exact dataset source contents."""

    payload = {
        "recipe": input_mapping_generation_params(mapping_cfg),
        "sources": source_fingerprints or input_mapping_source_fingerprints(cfg),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)[:16]


def input_mapping_source_fingerprints(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, configured_path in (
        ("documents", cfg.dataset.documents_path),
        ("queries", cfg.dataset.queries_path),
        ("qrels", cfg.dataset.qrels_path),
    ):
        path = project_path(configured_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset {name} file does not exist: {path}")
        fingerprints[name] = {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return fingerprints


def materialized_mapping_path(
    cfg: DictConfig,
    mapping_cfg: DictConfig,
    *,
    source_fingerprints: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Return the content-addressed reusable path for a generated mapping."""

    cache_key = input_mapping_cache_key(
        cfg,
        mapping_cfg,
        source_fingerprints=source_fingerprints,
    )
    name = str(mapping_cfg.get("name", "generated"))
    file_stem = name.replace("\\", "/").replace("/", "__")
    filename = f"{file_stem}.{cache_key}.json"
    return project_path(cfg.paths.input_mappings_dir) / str(cfg.dataset.name) / filename


def prepare_generated_input_mapping(cfg: DictConfig) -> tuple[GeneratedInputMapping, Path, bool]:
    """Generate or reuse a content-addressed input mapping for later inference runs."""

    mapping_cfg = cfg.get("input_mapping")
    mapping_type = str(mapping_cfg.get("type", "")) if mapping_cfg else ""
    if mapping_type != "generated":
        raise ValueError("prepare_mapping requires an input_mapping config with type: generated.")

    source_fingerprints = input_mapping_source_fingerprints(cfg)
    mapping_path = materialized_mapping_path(
        cfg,
        mapping_cfg,
        source_fingerprints=source_fingerprints,
    )
    metadata_path = metadata_path_for(mapping_path)
    if mapping_path.is_file() and metadata_path.is_file():
        return (
            GeneratedInputMapping(
                mapping=read_json(mapping_path),
                metadata=read_json(metadata_path),
            ),
            mapping_path,
            True,
        )
    if mapping_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Input mapping cache is incomplete; expected both {mapping_path} and {metadata_path}."
        )

    generated = generate_input_mapping(
        dataset_name=str(cfg.dataset.name),
        documents=read_jsonl(cfg.dataset.documents_path),
        queries=read_jsonl(cfg.dataset.queries_path),
        qrels=read_jsonl(cfg.dataset.qrels_path),
        **input_mapping_generation_params(mapping_cfg),
    )
    generated.metadata["mapping_name"] = str(mapping_cfg.get("name", "generated"))
    generated.metadata["recipe_hash"] = input_mapping_recipe_hash(mapping_cfg)
    generated.metadata["cache_key"] = input_mapping_cache_key(
        cfg,
        mapping_cfg,
        source_fingerprints=source_fingerprints,
    )
    generated.metadata["recipe"] = input_mapping_generation_params(mapping_cfg)
    generated.metadata["sources"] = source_fingerprints
    write_generated_mapping(generated, mapping_path=mapping_path, overwrite=False)
    return generated, mapping_path, False


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
    mapping_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write mapping JSON and sidecar metadata."""

    metadata_path = metadata_path_for(mapping_path)
    if not overwrite:
        for path in (mapping_path, metadata_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    write_json_atomic(mapping_path, generated.mapping)
    write_json_atomic(metadata_path, generated.metadata)
    return mapping_path, metadata_path


def metadata_path_for(mapping_path: Path) -> Path:
    return mapping_path.with_name(f"{mapping_path.stem}.meta.json")


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
