# Experimental Haystack components

This project evaluates third-party Haystack integrations without promoting them into the
shared `retrieval-components` package. Its reusable configuration fragments cover:

- FastEmbed dense document and query embedders;
- FastEmbed sparse document and query embedders;
- FastEmbed cross-encoder and late-interaction rankers; and
- Chonkie token, sentence, recursive, and semantic document splitters.

The project contains three SciFact experiment suites:

| Experiment | Question |
| --- | --- |
| `fastembed-dense-scifact` | Does FastEmbed preserve the effectiveness of an equivalent BGE-small SentenceTransformers pipeline? |
| `fastembed-rankers-scifact` | How do FastEmbed cross-encoder and late-interaction rankers compare with the repository's SentenceTransformers reranker over a fixed candidate set? |
| `chonkie-chunkers-scifact` | How do Chonkie's chunking strategies compare with the current LangChain recursive-character baseline? |

FastEmbed's sparse embedders are configuration- and construction-tested, but intentionally do
not have a retrieval experiment yet. The repository's JSONL index serializes dense embeddings
only; a sparse experiment needs a sparse-capable document store or a deliberately extended index
contract.

Chonkie emits the original document id as `meta.source_id`. The project-local
`SourceDocumentIdAdapter` copies that value to the repository's expected
`meta.source_document_id` field while preserving the rest of each Haystack `Document`. This is a
contract adapter, not a replacement chunker.

## Setup and validation

From this directory:

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run prepare-beir --data-dir ../../data --dataset scifact
```

Each experiment card lists its run entrypoints and execution order. Run a single entrypoint with:

```powershell
uv run stage --entrypoint experiments/<experiment>/configs/runs/<run>.yaml
```

Model-backed components download their checkpoints on first execution. Unit tests only compose
and deserialize pipelines; they do not warm models or make network requests.

`chonkie-haystack` 1.0.0 declares `chonkie[all]`, although these components only require Chonkie's
base chunkers plus `model2vec` for semantic splitting. The project uses a local uv dependency
override to omit unrelated API/server extras that conflict with the repository's pinned
SentenceTransformers stack. The integration itself remains the unmodified published package.
