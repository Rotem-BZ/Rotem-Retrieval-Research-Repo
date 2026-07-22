---
experiment_id: fastembed-dense-scifact
status: draft
project: experimental-components
created: 2026-07-23
---

# FastEmbed dense embedding parity on SciFact

## Question and hypothesis

Does FastEmbed's ONNX-backed `BAAI/bge-small-en-v1.5` integration preserve retrieval
effectiveness relative to Haystack's SentenceTransformers integration for the same checkpoint?
The proposed hypothesis is that the absolute `NDCG@10` difference is at most `0.005` when the
dataset, prefixes, cosine similarity, retrieval depth, and model checkpoint are fixed.

## Arms

| Arm | Indexing and query embedding implementation | Index id |
| --- | --- | --- |
| Baseline | Haystack SentenceTransformers | `fastembed-dense-scifact-st` |
| Treatment | `fastembed-haystack` | `fastembed-dense-scifact-fastembed` |

Both arms use the full BEIR SciFact test split and retrieve 100 documents per query. Evaluate
`baseline-inference` and `fastembed-inference` with their matching evaluation entrypoints. Compare
resolved configs before interpreting results; the intended differences are the embedder classes
and the separate index ids only.

## Execution order

Run these entrypoints from the project directory:

1. `configs/runs/baseline-indexing.yaml`
2. `configs/runs/fastembed-indexing.yaml`
3. `configs/runs/baseline-inference.yaml`
4. `configs/runs/fastembed-inference.yaml`
5. `configs/runs/baseline-evaluation.yaml`
6. `configs/runs/fastembed-evaluation.yaml`

Report all configured metrics, with `NDCG@10` primary. Also compare indexing and inference elapsed
times from manifests, but treat timing as descriptive unless hardware, caches, process state, and
thread counts are controlled in a dedicated benchmark.

