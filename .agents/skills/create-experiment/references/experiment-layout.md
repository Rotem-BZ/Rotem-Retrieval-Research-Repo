# Retrieval experiment layout

Create experiments below `<project>/experiments/<experiment-slug>/`:

```text
<experiment-slug>/
|-- experiment.md
|-- analysis.ipynb
`-- configs/
    |-- base-experiment-configs/
    |   |-- inference.yaml
    |   `-- evaluation.yaml
    `-- runs/
        |-- baseline.yaml
        |-- treatment.yaml
        |-- baseline-evaluation.yaml
        `-- treatment-evaluation.yaml
```

Add indexing base and run files only when producing a new index is part of the experiment.

## Experiment card

```markdown
# <Experiment name>

## Experiment

<Brief description.>

## Hypothesis

<The user's exact hypothesis.>

## Runs

| Run | Stage | Difference |
| --- | --- | --- |
| `baseline` | Inference | Uses `<baseline pipeline>`. |
| `treatment` | Inference | Replaces the pipeline with `<treatment pipeline>`. |
| `baseline-evaluation` | Evaluation | Evaluates `<experiment>--baseline`. |
| `treatment-evaluation` | Evaluation | Evaluates `<experiment>--treatment`. |
```

Keep the card to these three sections.

## Complete shared bases

The base owns every shared setting needed to compose a runnable stage. A typical inference base is:

```yaml
defaults:
  - /stages/inference
  - override /dataset: <dataset choice>
  - override /pipeline/inference@pipeline: <baseline pipeline choice>
  - override /selections/embedding_model@selections.embedding_model: <embedding-model choice>
  - override /runtime: <runtime choice>
  - _self_

selections:
  index_id: <exact existing index id>

pipeline:
  components:
    retriever:
      init_parameters:
        top_k: <retrieval depth>
```

A typical evaluation base is:

```yaml
defaults:
  - /stages/evaluation
  - override /dataset: <dataset choice>
  - _self_

metrics:
  - <target metric>
```

Remove non-applicable fields and add any mandatory fields revealed by Hydra composition.

## Minimal run entrypoints

Run configs express only identity through their filename and true experimental differences:

```yaml
# baseline.yaml
defaults:
  - /base-experiment-configs/inference
  - _self_
```

```yaml
# treatment.yaml
defaults:
  - /base-experiment-configs/inference
  - override /pipeline/inference@pipeline: <treatment pipeline choice>
  - _self_
```

```yaml
# baseline-evaluation.yaml
defaults:
  - /base-experiment-configs/evaluation
  - _self_

stage:
  inference_run_id: <experiment slug>--baseline
```

Use the analogous exact treatment run ID for treatment evaluation. Never add launcher-controlled `stage.run_id`, `paths.project_root`, or `experiment.*` values.
