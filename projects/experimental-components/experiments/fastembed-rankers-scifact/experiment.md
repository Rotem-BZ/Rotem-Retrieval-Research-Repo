---
experiment_id: fastembed-rankers-scifact
status: draft
project: experimental-components
created: 2026-07-23
---

# FastEmbed rerankers on fixed SciFact candidates

## Question and hypothesis

How do FastEmbed's cross-encoder and late-interaction rankers compare with Haystack's
SentenceTransformers cross-encoder when all arms rerank candidates from the same E5-small index?
The proposed primary hypothesis is that at least one FastEmbed arm improves `NDCG@10` over the
SentenceTransformers baseline without reducing `Recall@100`.

## Arms and controls

| Arm | Ranker |
| --- | --- |
| Baseline | `SentenceTransformersSimilarityRanker` with MS MARCO MiniLM-L6-v2 |
| Treatment 1 | `FastembedRanker` with the Xenova conversion of MS MARCO MiniLM-L6-v2 |
| Treatment 2 | `FastembedLateInteractionRanker` with ColBERT v2 |

First run `candidate-indexing`. Every inference arm then uses that exact
`fastembed-rankers-scifact-e5-candidates` index, the same `intfloat/e5-small-v2` query embedder,
and dense retrieval depth 100. Rankers retain 100 results so the ranker comparison does not
artificially cap `Recall@100`.

## Execution order

1. `configs/runs/candidate-indexing.yaml`
2. `configs/runs/sentence-transformers-inference.yaml`
3. `configs/runs/fastembed-cross-encoder-inference.yaml`
4. `configs/runs/fastembed-late-interaction-inference.yaml`
5. Run the three matching `*-evaluation.yaml` entrypoints.

`NDCG@10` is primary; `MRR@50` and recall metrics are secondary. This is an implementation-level
comparison, not a backend-only parity test: the late-interaction arm uses a different model
family, and the Xenova conversion may differ numerically from the baseline checkpoint.

