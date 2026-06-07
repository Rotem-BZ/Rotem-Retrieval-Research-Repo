# Retrieval Research

A small research framework for information retrieval experiments with Hydra-managed
configuration and Haystack `AsyncPipeline` execution.

The initial scaffold is intentionally simple: it includes file-backed JSONL
indexing, a toy keyword retriever, and recall/MRR evaluation so that the
research workflow can be exercised before production-grade components are added.

Install and run with uv:

```bash
uv sync --extra dev
uv run rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
uv run rr inference dataset=toy pipeline/inference@pipeline=dummy_keyword
uv run rr evaluation dataset=toy
```

See [docs/research_workflows.md](docs/research_workflows.md) for the workflow.
