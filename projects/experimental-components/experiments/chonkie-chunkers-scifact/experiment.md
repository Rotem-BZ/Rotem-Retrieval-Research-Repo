---
experiment_id: chonkie-chunkers-scifact
status: draft
project: experimental-components
created: 2026-07-23
---

# Chonkie chunking strategies on SciFact

## Question and hypothesis

Do Chonkie's boundary-aware chunkers improve dense chunk retrieval over the repository's current
LangChain recursive-character splitter? The proposed hypothesis is that at least one Chonkie arm
improves `NDCG@10` over the baseline while preserving or improving `Recall@100`.

## Arms

| Arm | Splitter | Nominal settings |
| --- | --- | --- |
| Baseline | LangChain recursive character | 360 characters, 60 overlap |
| Token | Chonkie token | character tokenizer, 360 size, 60 overlap |
| Sentence | Chonkie sentence | character tokenizer, 360 size, 60 overlap |
| Recursive | Chonkie recursive | character tokenizer, 360 size |
| Semantic | Chonkie semantic | Potion-32M, 360 target size, threshold 0.8 |

All arms use the full SciFact corpus, `intfloat/e5-small-v2`, identical E5 prefixes, cosine
similarity, and retrieval depth 100. Each splitter receives its own immutable index. Chonkie's
recursive and semantic APIs do not expose the baseline's overlap control, so this is a strategy
comparison rather than a single-variable overlap test.

## Execution order

For each arm (`baseline`, `token`, `sentence`, `recursive`, `semantic`), run its `*-indexing.yaml`,
then `*-inference.yaml`, then `*-evaluation.yaml` entrypoint. Indexing can be parallelized only if
memory and model-download contention are acceptable.

`NDCG@10` is primary. Record chunk count, character-length distribution, and source-document
coverage for each index before interpreting retrieval metrics. The semantic arm has an additional
model and should be evaluated separately for indexing cost. A follow-up experiment should tune
chunk sizes on a development dataset rather than selecting them after viewing SciFact results.

