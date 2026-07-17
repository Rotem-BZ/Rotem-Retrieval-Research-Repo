"""Candidate input mappings for inference runs."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from haystack import Document
from omegaconf import DictConfig

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

    def candidate_ids(self, query_id: str) -> list[str]:
        if query_id in self.candidate_ids_by_query:
            return self.candidate_ids_by_query[query_id]
        if self.default_candidate_ids is not None:
            return self.default_candidate_ids
        raise KeyError(f"No candidate ids configured for query {query_id!r}.")

    def candidate_documents(self, query_id: str) -> list[Document]:
        return [self.documents_by_id[document_id] for document_id in self.candidate_ids(query_id)]


@dataclass(frozen=True)
class GeneratedInputMapping:
    """Generated mapping payload plus sidecar metadata."""

    mapping: dict[str, list[str]]
    metadata: dict[str, Any]


def resolve_inference_mapping(cfg: DictConfig) -> InferenceMapping:
    """Resolve configured inference candidates from dataset files and input_mapping config."""

    documents = read_jsonl(cfg.dataset.documents_path)
    queries = read_jsonl(cfg.dataset.queries_path)
    documents_by_id = _documents_by_id(documents)
    queries_by_id = {str(query["id"]): query for query in queries}
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
            queries_by_id=queries_by_id,
            documents_by_id=documents_by_id,
        )

    if mapping_type != "file":
        raise ValueError(f"Unsupported input mapping type: {mapping_type!r}")

    return _resolve_file_mapping(
        project_path(mapping_cfg.path),
        mapping_cfg=mapping_cfg,
        cfg=cfg,
        queries_by_id=queries_by_id,
        documents_by_id=documents_by_id,
    )


def validate_input_mapping_config(
    cfg: DictConfig,
    *,
    require_prepared: bool = False,
    require_generated: bool = False,
) -> Path | None:
    """Validate input-mapping selection and return its materialized path when applicable."""

    mapping_cfg = cfg.get("input_mapping")
    mapping_type = str(mapping_cfg.get("type", "full_dataset")) if mapping_cfg else "full_dataset"
    if require_generated and mapping_type != "generated":
        raise ValueError("This stage requires an input_mapping config with type: generated.")
    if mapping_type == "full_dataset":
        return None
    if mapping_type == "file":
        path = project_path(mapping_cfg.path)
    elif mapping_type == "generated":
        path = materialized_mapping_path(cfg, mapping_cfg)
    else:
        raise ValueError(f"Unsupported input mapping type: {mapping_type!r}")

    if require_prepared and not path.is_file():
        raise FileNotFoundError(f"Input mapping is not prepared: {path}")
    return path


def _resolve_file_mapping(
    mapping_path: Path,
    *,
    mapping_cfg: DictConfig,
    cfg: DictConfig,
    queries_by_id: dict[str, dict[str, Any]],
    documents_by_id: dict[str, Document],
) -> InferenceMapping:
    configured_dataset = mapping_cfg.get("dataset")
    if configured_dataset is not None and str(configured_dataset) != str(cfg.dataset.name):
        raise ValueError(
            "Input mapping dataset does not match selected dataset: "
            f"{configured_dataset!r} != {cfg.dataset.name!r}."
        )

    raw_mapping = read_json(mapping_path)
    candidate_ids_by_query = _validate_input_mapping(
        raw_mapping,
        queries_by_id=queries_by_id,
        documents_by_id=documents_by_id,
    )
    selected_queries = [queries_by_id[query_id] for query_id in candidate_ids_by_query]

    return InferenceMapping(
        queries=selected_queries,
        candidate_ids_by_query=candidate_ids_by_query,
        documents_by_id=documents_by_id,
    )


def input_mapping_generation_params(mapping_cfg: DictConfig) -> dict[str, Any]:
    """Return the generation parameters represented by an input-mapping recipe."""

    params = {
        "seed": int(mapping_cfg.get("seed", 0)),
        "query_subset_size": _optional_int(mapping_cfg.get("query_subset_size")),
        "document_subset_size": _optional_int(mapping_cfg.get("document_subset_size")),
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
    filename = f"{_mapping_file_stem(name)}.{cache_key}.json"
    return project_path(cfg.paths.input_mappings_dir) / str(cfg.dataset.name) / filename


def prepare_generated_input_mapping(
    cfg: DictConfig, *, persist: bool = True
) -> tuple[GeneratedInputMapping, Path, bool]:
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
    if persist:
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

    _validate_non_negative(
        random_docs_per_query=random_docs_per_query,
        easy_negative_docs_per_query=easy_negative_docs_per_query,
        gold_passage_docs_per_query=gold_passage_docs_per_query,
    )
    rng = random.Random(seed)
    document_ids = [str(document["id"]) for document in documents]
    query_ids = [str(query["id"]) for query in queries]
    selected_query_ids = _sample_subset(query_ids, query_subset_size, rng, "queries")

    annotated_by_query = _annotated_by_query(qrels)
    positive_by_query = _positive_by_query(qrels)
    required_document_ids = set().union(
        *(annotated_by_query.get(query_id, set()) for query_id in selected_query_ids)
    )
    active_document_ids = _active_document_ids(
        document_ids,
        required_document_ids=required_document_ids,
        document_subset_size=document_subset_size,
        rng=rng,
    )
    active_document_set = set(active_document_ids)

    annotated_anywhere = set().union(*annotated_by_query.values()) if annotated_by_query else set()
    easy_negative_pool = [
        document_id for document_id in active_document_ids if document_id not in annotated_anywhere
    ]
    if easy_negative_docs_per_query and not easy_negative_pool:
        raise ValueError("No easy negative documents exist for this dataset and document subset.")

    mapping: dict[str, list[str]] = {}
    for query_id in selected_query_ids:
        included = [
            document_id
            for document_id in document_ids
            if document_id in annotated_by_query.get(query_id, set())
        ]
        missing_required = [
            document_id for document_id in included if document_id not in active_document_set
        ]
        if missing_required:
            raise ValueError(
                f"Document subset excludes annotated documents for query {query_id}: "
                f"{missing_required}"
            )

        included_set = set(included)
        included.extend(
            _sample_new(
                active_document_ids,
                excluded=included_set,
                count=random_docs_per_query,
                rng=rng,
                label=f"random documents for query {query_id}",
            )
        )
        included_set = set(included)
        included.extend(
            _sample_new(
                easy_negative_pool,
                excluded=included_set,
                count=easy_negative_docs_per_query,
                rng=rng,
                label=f"easy negative documents for query {query_id}",
            )
        )
        included_set = set(included)
        gold_pool = _gold_negative_pool(
            query_id,
            positive_by_query=positive_by_query,
            annotated_by_query=annotated_by_query,
            active_document_ids=active_document_ids,
        )
        included.extend(
            _sample_new(
                gold_pool,
                excluded=included_set,
                count=gold_passage_docs_per_query,
                rng=rng,
                label=f"gold passage negative documents for query {query_id}",
            )
        )
        mapping[query_id] = included

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


def _documents_by_id(records: Iterable[dict[str, Any]]) -> dict[str, Document]:
    documents_by_id: dict[str, Document] = {}
    for record in records:
        document = Document(
            id=record.get("id"),
            content=record.get("content", ""),
            meta=dict(record.get("meta") or {}),
            score=record.get("score"),
            embedding=record.get("embedding"),
        )
        if document.id is None:
            raise ValueError(f"Document is missing an id: {record}")
        document_id = str(document.id)
        if document_id in documents_by_id:
            raise ValueError(f"Duplicate document id in dataset: {document_id}")
        documents_by_id[document_id] = document
    return documents_by_id


def _validate_input_mapping(
    raw_mapping: Any,
    *,
    queries_by_id: dict[str, dict[str, Any]],
    documents_by_id: dict[str, Document],
) -> dict[str, list[str]]:
    if not isinstance(raw_mapping, dict):
        raise ValueError("Input mapping JSON must be an object keyed by query id.")

    mapping: dict[str, list[str]] = {}
    for raw_query_id, raw_candidate_ids in raw_mapping.items():
        query_id = str(raw_query_id)
        if query_id not in queries_by_id:
            raise ValueError(f"Input mapping references unknown query id: {query_id}")
        if not isinstance(raw_candidate_ids, list):
            raise ValueError(f"Input mapping for query {query_id} must be a list.")

        candidate_ids = [str(document_id) for document_id in raw_candidate_ids]
        missing_document_ids = [
            document_id for document_id in candidate_ids if document_id not in documents_by_id
        ]
        if missing_document_ids:
            raise ValueError(
                f"Input mapping for query {query_id} references unknown document ids: "
                f"{missing_document_ids}"
            )
        mapping[query_id] = list(dict.fromkeys(candidate_ids))

    return mapping


def _validate_non_negative(**values: int) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")


def _sample_subset(
    ids: list[str],
    subset_size: int | None,
    rng: random.Random,
    label: str,
) -> list[str]:
    if subset_size is None:
        return ids
    if subset_size < 0:
        raise ValueError(f"{label} subset size must be non-negative.")
    if subset_size > len(ids):
        raise ValueError(f"Requested {subset_size} {label}, but only {len(ids)} exist.")
    return sorted(rng.sample(ids, subset_size), key=ids.index)


def _active_document_ids(
    document_ids: list[str],
    *,
    required_document_ids: set[str],
    document_subset_size: int | None,
    rng: random.Random,
) -> list[str]:
    if document_subset_size is None:
        return document_ids
    if document_subset_size < 0:
        raise ValueError("document_subset_size must be non-negative.")
    if document_subset_size < len(required_document_ids):
        raise ValueError(
            "document_subset_size is smaller than the number of annotated documents required "
            "for the selected queries."
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
    return [document_id for document_id in document_ids if document_id in sampled]


def _annotated_by_query(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
    annotated: dict[str, set[str]] = {}
    for qrel in qrels:
        annotated.setdefault(str(qrel["query_id"]), set()).add(str(qrel["document_id"]))
    return annotated


def _positive_by_query(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
    positive: dict[str, set[str]] = {}
    for qrel in qrels:
        if int(qrel.get("relevance", 1)) > 0:
            positive.setdefault(str(qrel["query_id"]), set()).add(str(qrel["document_id"]))
    return positive


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


def _gold_negative_pool(
    query_id: str,
    *,
    positive_by_query: dict[str, set[str]],
    annotated_by_query: dict[str, set[str]],
    active_document_ids: list[str],
) -> list[str]:
    current_annotations = annotated_by_query.get(query_id, set())
    positive_elsewhere = (
        set().union(
            *(
                document_ids
                for other_query_id, document_ids in positive_by_query.items()
                if other_query_id != query_id
            )
        )
        if positive_by_query
        else set()
    )
    return [
        document_id
        for document_id in active_document_ids
        if document_id in positive_elsewhere and document_id not in current_annotations
    ]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _mapping_file_stem(name: str) -> str:
    return name.replace("\\", "/").replace("/", "__")
