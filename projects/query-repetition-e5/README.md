# Query repetition with E5-small

This project tests one deliberately small change: repeat each raw query twice before
the standard E5 query prefix is added, then compare it with the unchanged dense
`intfloat/e5-small-v2` pipeline.

The motivation is related to [Prompt Repetition Improves Non-Reasoning
LLMs](https://arxiv.org/abs/2512.14982) and, more directly, [Repetition Improves
Language Model Embeddings](https://arxiv.org/abs/2402.15449). This project is an
exploratory transfer test, not a reproduction: those explanations concern causal
language models, whereas E5-small is a bidirectional encoder. A gain is therefore an
empirical question.

## Package isolation

The project owns its environment and lockfile. Its `pyproject.toml` declares the
component contract as `retrieval-components==0.1.0`, while `[tool.uv.sources]`
resolves both monorepo dependencies locally and editably:

```toml
[tool.uv.sources]
retrieval-core = { path = "../../packages/retrieval-core", editable = true }
retrieval-components = { path = "../../packages/retrieval-components", editable = true }
```

The run manifest records the installed versions of both distributions. When the
component library is published, remove its `tool.uv.sources` entry to test the same
declared version from the package index.

## Run the comparison

From this directory:

```powershell
uv sync --extra dev
./scripts/run_experiment.ps1
```

For a CUDA 12.6 PyTorch environment, use `uv sync --extra dev --extra torch-cu126`.
The experiment defaults to CPU; run `./scripts/run_experiment.ps1 -Device cuda` to
use a configured CUDA environment.
The script downloads and converts BEIR SciFact, validates both pipeline graphs,
creates one shared E5-small index, runs baseline and repeated-query inference against
that exact index, evaluates both runs, and prints per-metric deltas.

To run only the repeated-query pipeline directly on Windows or Linux, prepare the
dataset and index once, then launch inference from this project directory:

```shell
uv run prepare-beir --data-dir data --dataset scifact
uv run stage indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.run_id=e5-small-index
uv run stage inference dataset=beir_scifact pipeline/inference@pipeline=dense_query_repetition selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.indexing_run_id=e5-small-index runtime.query_concurrency_limit=8
```

Indexing run IDs are immutable. Change `e5-small-index` before repeating the indexing
command, and pass the same new ID to `stage.indexing_run_id` during inference.

The important comparison is the sign and size of the delta, especially for
`NDCG@10`, `Recall@10`, and `MRR@50`. Because this is one dataset and one run, treat
small differences as a prompt for broader evaluation rather than a general result.
