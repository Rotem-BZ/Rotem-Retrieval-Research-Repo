# Query repetition with E5-small

This project tests one deliberately small change: repeat each raw query twice before
the standard E5 query prefix is added, then compare it with the unchanged dense
`intfloat/e5-small-v2` pipeline.

Its project-owned Hydra pipeline is selected as
`query_repetition/dense_query_repetition`; unqualified pipeline choices come from
`retrieval-core`.

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

## Run the experiment

From this directory:

```powershell
uv sync --extra dev
```

For a CUDA 12.6 PyTorch environment, use `uv sync --extra dev --extra torch-cu126`.
On an older NVIDIA driver that requires PyTorch 2.5.1 with CUDA 12.4, use
`uv sync --extra dev --extra torch-cu124-legacy`. These PyTorch extras are mutually
exclusive. Use the legacy extra with a Python version supported by PyTorch 2.5.1,
such as Python 3.12; PyTorch 2.5.1 does not provide Python 3.14 wheels. The experiment
defaults to CPU. Use checked-in experiment entrypoints for canonical result runs;
ad hoc terminal overrides are for development only.

The durable experiment workspace is
[`experiments/query-repetition-e5-small-scifact`](experiments/query-repetition-e5-small-scifact).
It contains the research card, a complete experiment base config, minimal run layers,
and an analysis notebook; the reusable repetition pipeline lives in the project's
`configs/` tree. After creating the
shared index, update its exact ID in
[`base-experiment-configs/inference.yaml`](experiments/query-repetition-e5-small-scifact/configs/base-experiment-configs/inference.yaml).
On Linux,
`uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py` lets you choose the
experiment and the subset of runs to launch in GNU Screen. Use
`uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/query-repetition-e5-small-scifact`
to add another run interactively.

After preparing the dataset and shared index described by the experiment card, launch
the checked-in inference runs, then their checked-in evaluation runs:

```powershell
uv run stage inference --entrypoint experiments/query-repetition-e5-small-scifact/configs/runs/baseline.yaml
uv run stage inference --entrypoint experiments/query-repetition-e5-small-scifact/configs/runs/repeated.yaml
uv run stage evaluation --entrypoint experiments/query-repetition-e5-small-scifact/configs/runs/baseline-evaluation.yaml
uv run stage evaluation --entrypoint experiments/query-repetition-e5-small-scifact/configs/runs/repeated-evaluation.yaml
```

To run only the repeated-query pipeline directly on Windows or Linux, prepare the
dataset and index once, then launch inference from this project directory:

```shell
uv run prepare-beir --data-dir data --dataset scifact
uv run stage indexing dataset=beir_scifact runtime=cpu pipeline/indexing@pipeline=dense/documents_in_memory selections/embedding_model=e5/small_v2 selections.index_id=e5-small-index stage.run_id=e5-small-indexing
uv run stage inference dataset=beir_scifact runtime=cpu pipeline/inference@pipeline=query_repetition/dense_query_repetition selections/embedding_model=e5/small_v2 selections.index_id=e5-small-index runtime.query_concurrency_limit=8
```

Index IDs are immutable. Change `e5-small-index` before repeating the indexing command,
and pass the same new ID as `selections.index_id` during inference.

The important comparison is the sign and size of the delta, especially for
`NDCG@10`, `Recall@10`, and `MRR@50`. Because this is one dataset and one run, treat
small differences as a prompt for broader evaluation rather than a general result.

## Analyze predictions

Open the experiment's
[`analysis.ipynb`](experiments/query-repetition-e5-small-scifact/analysis.ipynb).
Add exact IDs to `INFERENCE_RUNS` and `EVALUATION_RUNS`, then run the cells. Shared
`retrieval_core.utils.analysis` helpers resolve artifacts and build:

- `predictions_df`, one row per retrieved result with rank, score, content, metadata,
  and matched qrel relevance; and
- `query_summary_df`, one row per run and query with retrieval depth, relevant counts,
  first relevant rank, reciprocal rank, recall, and query-length fields.

The notebook also presents the target metric comparison and common aggregate and
per-query plots. Export the readable final report without modifying notebook outputs:

```powershell
uv run python ../../awesome-dev-tools/export_experiment_report.py experiments/query-repetition-e5-small-scifact
```

For a small executable project run over the checked-in toy fixture, see
[`query-repetition-e5-small-toy`](experiments/query-repetition-e5-small-toy/experiment.md)
and the repository's [example command guide](../../docs/example_commands.md).
