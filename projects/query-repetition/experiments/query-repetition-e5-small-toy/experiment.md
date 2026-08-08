# Query repetition on the toy dataset

## Experiment

Compare the unchanged `intfloat/e5-small-v2` query pipeline with the repeated-query
pipeline on the repository's checked-in toy retrieval dataset.

## Hypothesis

Both the repeated-query treatment and the baseline should have perfect recall.

## Runs

| Run | Stage | Difference |
| --- | --- | --- |
| `repeated` | Inference | Replaces the base inference pipeline with `query_repetition/dense_query_repetition`, which repeats each query before E5 preprocessing. |

No baseline or evaluation run is currently checked in for the toy experiment.
