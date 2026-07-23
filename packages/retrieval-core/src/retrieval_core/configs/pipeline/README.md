# Pipeline choices

Pipeline choices are organized first by stage and then by intent. Select them
with Hydra's existing `pipeline/<stage>@pipeline=<choice>` override.

| Stage | Choice | Use when | Required selections |
| --- | --- | --- | --- |
| Indexing | `dense/documents_jsonl` | Embed and persist whole documents in JSONL. | `index_id`, `embedding_model` |
| Indexing | `dense/chunks_jsonl` | Split, embed, and persist document chunks in JSONL. | `index_id`, `embedding_model` |
| Inference | `retrieve/dense_jsonl` | Retrieve from either supported dense JSONL index granularity. | `index_id`, `embedding_model` |
| Inference | `retrieve/hybrid_rrf_jsonl` | Fuse keyword and dense retrieval with classic equal-weight RRF. | `index_id`, `embedding_model` |
| Inference | `rerank/bi_encoder` | Rerank a materialized candidate set with embedding similarity. | `embedding_model` |
| Inference | `rerank/cross_encoder` | Rerank a materialized candidate set with a cross-encoder. | `reranker_model` |

The choices under `scaffold/` are minimal, inexpensive pipelines for CLI,
configuration, and end-to-end validation. They are not recommended experiment
baselines.

Fusion algorithms are component fragments rather than incomplete pipeline
templates. For example, `retrieve/hybrid_rrf_jsonl` selects
`/component/fusion@components.fusion: rrf` and connects both ranked-list
producers.
