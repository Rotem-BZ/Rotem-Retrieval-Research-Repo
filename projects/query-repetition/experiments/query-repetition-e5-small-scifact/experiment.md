# Repeating E5-small queries on BEIR SciFact

## Experiment

Compare dense retrieval on the BEIR SciFact test split using the unchanged
`intfloat/e5-small-v2` query pipeline as the baseline and a treatment that repeats
each raw query twice before standard E5 preprocessing.

## Hypothesis

The repeated-query treatment should have better recall than the baseline.

## Runs

| Run | Stage | Difference |
| --- | --- | --- |
| `baseline` | Inference | Uses the shared experiment configuration unchanged. |
| `repeated` | Inference | Replaces the baseline inference pipeline with `query_repetition/dense_query_repetition`, which repeats each query before E5 preprocessing. |
| `baseline-evaluation` | Evaluation | Evaluates the exact `baseline` inference run. |
| `repeated-evaluation` | Evaluation | Evaluates the exact `repeated` inference run; otherwise uses the same evaluation configuration as `baseline-evaluation`. |
