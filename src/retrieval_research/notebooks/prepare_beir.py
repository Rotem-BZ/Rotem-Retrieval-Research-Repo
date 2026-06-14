"""Prepare BEIR datasets in a CCDS-style data layout.

Run this as a Python notebook script, cell by cell in an editor that supports
`# %%`, or as a regular script:

    uv run python src/retrieval_research/notebooks/prepare_beir.py
"""

# %%
from __future__ import annotations

import csv
import json
import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

try:
    import certifi
except ImportError:  # pragma: no cover - optional runtime convenience
    certifi = None

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

BEIR_DATASET_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}

DATASETS = [
    {
        "dataset_name": "scifact",
        "split": "test",
        "processed_name": "scifact",
        "max_documents": None,
        "max_queries": None,
    },
    {
        "dataset_name": "scifact",
        "split": "test",
        "processed_name": "scifact_smoke",
        "max_documents": 250,
        "max_queries": 25,
    },
]

# %%


def prepare_beir_dataset(
    *,
    dataset_name: str,
    split: str,
    processed_name: str | None = None,
    max_documents: int | None = None,
    max_queries: int | None = None,
    force_download: bool = False,
    force_extract: bool = False,
) -> dict[str, Any]:
    raw_dir = RAW_DIR / "beir" / dataset_name
    interim_dir = INTERIM_DIR / "beir" / dataset_name
    processed_dir = PROCESSED_DIR / "beir" / (processed_name or dataset_name)

    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    archive_path = raw_dir / f"{dataset_name}.zip"
    downloaded = download(BEIR_DATASET_URLS[dataset_name], archive_path, force=force_download)
    extracted_root = extract(archive_path, interim_dir, dataset_name, force=force_extract)

    result = process_beir_dataset(
        dataset_name=dataset_name,
        split=split,
        extracted_root=extracted_root,
        processed_dir=processed_dir,
        max_documents=max_documents,
        max_queries=max_queries,
    )
    return {
        "dataset_name": dataset_name,
        "split": split,
        "downloaded": downloaded,
        "archive_path": str(archive_path),
        "extracted_root": str(extracted_root),
        **result,
    }


def download(url: str, archive_path: Path, *, force: bool = False) -> bool:
    if archive_path.exists() and not force:
        return False

    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    with urllib.request.urlopen(url, context=context) as response, archive_path.open(
        "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)
    return True


def extract(archive_path: Path, interim_dir: Path, dataset_name: str, *, force: bool = False) -> Path:
    expected_root = interim_dir / dataset_name
    if expected_root.joinpath("corpus.jsonl").is_file() and not force:
        return expected_root

    with ZipFile(archive_path) as archive:
        archive.extractall(interim_dir)

    for candidate in [expected_root, interim_dir, *interim_dir.iterdir()]:
        if candidate.is_dir() and candidate.joinpath("corpus.jsonl").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find extracted BEIR root under {interim_dir}.")


# %%


def process_beir_dataset(
    *,
    dataset_name: str,
    split: str,
    extracted_root: Path,
    processed_dir: Path,
    max_documents: int | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    documents = list(document_records(extracted_root / "corpus.jsonl", dataset_name))
    queries = list(query_records(extracted_root / "queries.jsonl", dataset_name))
    qrels = list(qrel_records(extracted_root / "qrels" / f"{split}.tsv"))
    documents, queries, qrels = limit_records(
        documents=documents,
        queries=queries,
        qrels=qrels,
        max_documents=max_documents,
        max_queries=max_queries,
    )

    documents_path = write_jsonl(processed_dir / "documents.jsonl", documents)
    queries_path = write_jsonl(processed_dir / "queries.jsonl", queries)
    qrels_path = write_jsonl(processed_dir / "qrels.jsonl", qrels)
    input_mapping_path = write_json(processed_dir / "input_mapping.json", {})

    return {
        "documents_path": str(documents_path),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "input_mapping_path": str(input_mapping_path),
        "document_count": len(documents),
        "query_count": len(queries),
        "qrel_count": len(qrels),
        "max_documents": max_documents,
        "max_queries": max_queries,
    }


def document_records(path: Path, dataset_name: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            title = str(record.get("title") or "").strip()
            text = str(record.get("text") or "").strip()
            content = "\n".join(part for part in [title, text] if part)
            yield {
                "id": str(record["_id"]),
                "content": content,
                "meta": {
                    "dataset": dataset_name,
                    "title": title,
                    "beir_metadata": record.get("metadata") or {},
                },
            }


def query_records(path: Path, dataset_name: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            yield {
                "id": str(record["_id"]),
                "text": str(record.get("text") or ""),
                "meta": {
                    "dataset": dataset_name,
                    "beir_metadata": record.get("metadata") or {},
                },
            }


def qrel_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)

        if "query-id" in sample and "corpus-id" in sample:
            for row in csv.DictReader(handle, delimiter="\t"):
                yield {
                    "query_id": str(row["query-id"]),
                    "document_id": str(row["corpus-id"]),
                    "relevance": int(row["score"]),
                }
            return

        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) >= 4:
                query_id, document_id, relevance = row[0], row[2], row[3]
            else:
                query_id, document_id, relevance = row[0], row[1], row[2]
            yield {
                "query_id": str(query_id),
                "document_id": str(document_id),
                "relevance": int(relevance),
            }


# %%


def limit_records(
    *,
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    max_documents: int | None,
    max_queries: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if max_documents is None and max_queries is None:
        return documents, queries, qrels

    selected_query_ids = selected_query_ids_from_qrels(queries, qrels, max_queries)
    selected_qrels = [qrel for qrel in qrels if qrel["query_id"] in selected_query_ids]
    required_document_ids = {qrel["document_id"] for qrel in selected_qrels}
    limited_documents = select_documents(documents, required_document_ids, max_documents)
    available_document_ids = {document["id"] for document in limited_documents}

    return (
        limited_documents,
        [query for query in queries if query["id"] in selected_query_ids],
        [qrel for qrel in selected_qrels if qrel["document_id"] in available_document_ids],
    )


def selected_query_ids_from_qrels(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    max_queries: int | None,
) -> set[str]:
    if max_queries is None:
        return {query["id"] for query in queries}

    ordered_qrel_query_ids = list(dict.fromkeys(qrel["query_id"] for qrel in qrels))
    if ordered_qrel_query_ids:
        return set(ordered_qrel_query_ids[:max_queries])
    return {query["id"] for query in queries[:max_queries]}


def select_documents(
    documents: list[dict[str, Any]],
    required_document_ids: set[str],
    max_documents: int | None,
) -> list[dict[str, Any]]:
    if max_documents is None:
        return documents

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for document in documents:
        if document["id"] in required_document_ids:
            selected.append(document)
            seen.add(document["id"])

    for document in documents:
        if len(selected) >= max_documents:
            break
        if document["id"] not in seen:
            selected.append(document)
            seen.add(document["id"])

    return selected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


# %%
if __name__ == "__main__":
    for dataset in DATASETS:
        print(prepare_beir_dataset(**dataset))
