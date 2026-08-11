# `prepare_mapping` Stage

The `prepare_mapping` stage creates an immutable, reusable input mapping for inference.
An input mapping selects which queries inference runs and which candidate documents each
selected query receives.

## Inputs and outputs

The stage requires:

- a dataset containing documents, queries, and qrels;
- an `input_mapping_recipe` with `type: generated`;
- a unique, non-empty `stage.run_id`.

It writes:

```text
artifacts/input_mappings/<run-id>/input_mapping.json
artifacts/input_mappings/<run-id>/meta.json
```

`input_mapping.json` is an object keyed by the canonical query input field (`IN`). Each
value is the ordered list of candidate document ids for that query:

```json
{
  "q-1": ["doc-1", "doc-7", "doc-9"]
}
```

Only queries present as keys in the mapping are executed by inference. `meta.json` records
the recipe parameters and candidate-count summary.

## Recipe fields

Every generation parameter in the table below must be explicitly present in a stage
recipe. The stage does not supply omitted generation parameters.

| Parameter | Meaning |
| --- | --- |
| `seed` | Seeds the single random-number generator used for query selection, document selection, and per-query sampling. Because these operations share one generator, changing an earlier sampling setting can change later samples even when the seed is unchanged. |
| `query_subset_size` | `null` selects every query. A non-negative integer randomly selects exactly that many queries without replacement. `0` produces an empty mapping. Selected queries are emitted in their original dataset order. |
| `document_subset_size` | `null` makes every dataset document active. A non-negative integer randomly selects exactly that many documents without replacement, independently of qrels. `0` creates an empty active pool. Active documents are emitted in their original dataset order. The active pool is a hard boundary for all subsequent candidate selection. |
| `include_annotated_docs` | When `true`, automatically includes every active document having any qrel for the current query, including documents with label `0`. When `false`, qrels do not automatically insert candidates. Annotated documents can still be selected by another sampling rule. |
| `use_all_selected_documents_for_every_query` | When `true`, assigns the complete active document pool to every selected query and bypasses automatic annotated-document inclusion and all per-query sampling. Each per-query sampling count must then be `0` or `null`. |
| `random_docs_per_query` | Adds this many documents sampled from the active pool, excluding candidates already included for the current query. The pool is not filtered by qrels, so selected documents may be annotated or unannotated. |
| `easy_negative_docs_per_query` | Adds this many active documents having no qrel annotation for any query in the dataset, excluding candidates already included for the current query. |
| `gold_passage_docs_per_query` | Adds this many active documents with a positive qrel for another query. Documents having any qrel annotation for the current query, including label `0`, are excluded. Previously included candidates are also excluded. |

The three per-query sampling counts accept either a non-negative integer or `null`.
`null` and `0` both disable that sampling step. A negative count is invalid. Requesting
more documents than remain available in the applicable pool is also an error.

Recipes also contain these non-generation fields:

| Field | Meaning |
| --- | --- |
| `type` | Must be `generated`. |
| `name` | Human-readable recipe name recorded as `recipe_name` in `meta.json`. If omitted, the recorded name is `generated`. |
| `metadata.description` | Optional descriptive text for people reading the recipe; it does not affect generation. |

## Candidate construction order

Unless `use_all_selected_documents_for_every_query` is enabled, candidates are added for
each selected query in this order:

1. Annotated documents, when `include_annotated_docs` is `true`.
2. Random documents.
3. Easy negatives.
4. Gold-passage negatives.

Each step excludes candidates already added by an earlier step. All steps are restricted
to the active document pool established by `document_subset_size`.

## Complete recipe example

```yaml
metadata:
  description: Ten-query experiment over a random 100-document candidate pool.

type: generated
name: ten_queries_100_docs
seed: 13
query_subset_size: 10
document_subset_size: 100
include_annotated_docs: true
use_all_selected_documents_for_every_query: false
random_docs_per_query: 10
easy_negative_docs_per_query: 5
gold_passage_docs_per_query: 5
```

To give every selected query the same active document pool, use:

```yaml
metadata:
  description: Ten queries sharing one random 100-document pool.

type: generated
name: shared_100_docs
seed: 13
query_subset_size: 10
document_subset_size: 100
include_annotated_docs: true
use_all_selected_documents_for_every_query: true
random_docs_per_query: null
easy_negative_docs_per_query: null
gold_passage_docs_per_query: null
```

In this mode, `include_annotated_docs` has no effect on the resulting mapping because all
active documents are assigned to every selected query. It remains required as part of the
complete recipe schema.

Run the stage with a prepared recipe and a unique run id:

```powershell
uv run --project packages/retrieval-core stage prepare_mapping `
  dataset=beir_scifact `
  input_mapping_recipe=ten_queries_100_docs `
  stage.run_id=scifact_ten_queries_100_docs
```

Prepared mapping directories are immutable. Reusing an existing `stage.run_id` fails
rather than overwriting the previous mapping.
