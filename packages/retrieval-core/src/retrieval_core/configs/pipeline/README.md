# Pipeline choices

Pipeline choices are organized first by stage and then by intent. Select them
with Hydra's existing `pipeline/<stage>@pipeline=<choice>` override.

| Stage | Choice | Use when | Required selections |
| --- | --- | --- | --- |
| Indexing | `dense/documents_jsonl` | Embed and persist whole documents in JSONL. | `index_id`, `embedding_model` |
| Indexing | `dense/chunks_jsonl` | Split, embed, and persist document chunks in JSONL. | `index_id`, `embedding_model` |
| Inference | `retrieve/dense_jsonl` | Retrieve from either supported dense JSONL index granularity. | `index_id`, `embedding_model` |
| Inference | `rerank/bi_encoder` | Rerank a materialized candidate set with embedding similarity. | `embedding_model` |
| Inference | `rerank/cross_encoder` | Rerank a materialized candidate set with a cross-encoder. | `reranker_model` |

Fusion algorithms remain available as component fragments for project-owned
pipelines with multiple ranked-list producers.
