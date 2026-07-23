# Haystack v2 Components and Configuration

Use this reference for this repository's `haystack-ai>=2.30,<3` contract. Do not use Haystack 1.x node APIs, `BaseComponent`, `Pipeline.add_node`, or Hydra `_target_` syntax.

## Contents

1. [Minimal static component](#minimal-static-component)
2. [Python component contract](#python-component-contract)
3. [Dynamic and variadic sockets](#dynamic-and-variadic-sockets)
4. [Async, warm-up, and external systems](#async-warm-up-and-external-systems)
5. [Documents and serialization](#documents-and-serialization)
6. [Exports and import paths](#exports-and-import-paths)
7. [Component fragments](#component-fragments)
8. [Pipeline topology](#pipeline-topology)
9. [Hydra composition and placement](#hydra-composition-and-placement)
10. [Tests and validation](#tests-and-validation)
11. [Failure checklist](#failure-checklist)

## Minimal static component

Copy this shape for a component whose socket names do not depend on constructor configuration:

```python
"""Top-k document selection component."""

from __future__ import annotations

from haystack import Document, component


@component
class TopDocuments:
    """Keep the highest-scoring documents."""

    def __init__(self, top_k: int = 10) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        limit = self.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")
        ranked = sorted(
            documents,
            key=lambda document: float(document.score or 0.0),
            reverse=True,
        )
        return {"documents": ranked[:limit]}
```

The corresponding inline Haystack pipeline entry is:

```yaml
components:
  selector:
    type: retrieval_components.cascade.top_documents.TopDocuments
    init_parameters:
      top_k: 10
```

## Python component contract

### Class and constructor

- Import `component` from `haystack` and decorate the class with `@component`.
- Do not inherit from a Haystack base class.
- Keep `__init__` lightweight. Store configuration and validate invalid settings there; load models, indexes, clients, and files lazily in `warm_up()` or `run()`.
- Use YAML/JSON-safe constructor values: strings, numbers, booleans, `None`, lists, and dictionaries containing those values. Prefer a string path over a `Path`, and an import path or configuration dictionary over a live callable/client.
- Preserve the constructor values on the instance when they define component behavior. Haystack v2 automatically records ordinary `@component` constructor arguments for pipeline serialization.
- Ensure every YAML `init_parameters` key exactly matches a constructor parameter. Constructor defaults apply when a key is omitted.

### Input sockets

For static sockets, Haystack derives inputs from the annotated `run()` parameters:

```python
def run(self, query: str, documents: list[Document]) -> dict[str, list[Document]]:
    ...
```

This declares required `query` and `documents` inputs. A type union with `None` does not make a pipeline socket optional by itself. Give the parameter a default:

```python
def run(
    self,
    query: str,
    candidate_document_ids: list[str] | None = None,
) -> dict[str, list[Document]]:
    ...
```

Here `query` is required and `candidate_document_ids` is optional. Put required parameters before optional parameters and do not use unannotated socket parameters.

Do not use Haystack 1.x `set_input_type` patterns for ordinary static sockets. Use the `run()` signature.

### Output sockets and return values

Declare static outputs on `run()`:

```python
@component.output_types(documents=list[Document], rejected_documents=list[Document])
def run(self, documents: list[Document]) -> dict[str, list[Document]]:
    ...
    return {
        "documents": kept,
        "rejected_documents": rejected,
    }
```

- The decorator's keyword names are output socket names.
- The returned dictionary keys must exactly match the outputs emitted on that execution.
- Return a dictionary, not a bare value, tuple, dataclass, generator, or coroutine result from a synchronous `run()`.
- Use concrete socket types such as `str`, `list[float]`, and `list[Document]`. Socket compatibility is checked when the pipeline connects.
- An output may be left unconnected. A required input must be connected or provided as pipeline input at runtime.

## Dynamic and variadic sockets

Use dynamic named sockets only when constructor data determines the socket names. Weighted fusion is the repository pattern:

```python
from __future__ import annotations

from haystack import Document, component


@component
class WeightedFusion:
    def __init__(self, weights: dict[str, float]) -> None:
        if not weights:
            raise ValueError("weights must define at least one source")
        self.weights = dict(weights)
        component.set_input_types(
            self,
            **{source_name: list[Document] for source_name in self.weights},
        )
        component.set_output_types(self, documents=list[Document])

    def run(self, **ranked_lists: list[Document]) -> dict[str, list[Document]]:
        ...
        return {"documents": fused_documents}
```

Rules for dynamic sockets:

- The `run()` method must accept `**kwargs`; otherwise `component.set_input_type(s)` raises an error.
- Use `component.set_input_types(self, **mapping)` for required named inputs.
- Use `component.set_input_type(self, name, socket_type, default=value)` when one dynamic input needs a default and is therefore optional.
- Use `component.set_output_types` when outputs are also established on the instance.
- Do not combine `component.set_output_types` with `@component.output_types` on `run()` or `run_async()`.
- Make the config that controls dynamic names serializable. Each configured source name becomes a socket name and must match a connection receiver suffix.

A normal `list[Document]` socket receives one list payload. It is not a multi-sender socket. If multiple upstream components must connect to the same socket, Haystack v2 provides:

```python
from haystack.core.component.types import GreedyVariadic, Variadic
```

Use `Variadic[T]` only when the component should wait for all connected senders and receive their values as an iterable. Use `GreedyVariadic[T]` only when it should run as values arrive. For source-specific weights or names, prefer the repository's dynamic named-socket pattern because a variadic input loses the source name unless the payload carries it.

## Async, warm-up, and external systems

Every component must define synchronous `run()`. `AsyncPipeline` can execute synchronous components, so do not add async code without a real I/O or concurrency need.

When a component also defines `run_async()`:

- Declare it with `async def run_async(...)`.
- Give `run()` and `run_async()` exactly the same parameter names, annotations, order, and defaults.
- Give both methods exactly the same output socket declaration, or set instance output types once in `__init__`.
- Return the same dictionary shape from both methods.

```python
@component.output_types(text=str)
def run(self, text: str) -> dict[str, str]:
    return {"text": self._sync_transform(text)}

@component.output_types(text=str)
async def run_async(self, text: str) -> dict[str, str]:
    return {"text": await self._async_transform(text)}
```

Use an idempotent `warm_up()` for expensive reusable state such as a model or backend connection. Keep a sentinel so repeated calls do not initialize twice; Haystack does not track that for the component.

For external systems:

- Make network or filesystem behavior explicit in the class contract.
- Keep credentials and concrete model/runtime choices in config, not module globals.
- Represent client configuration with serializable constructor values. Create the client lazily, or resolve an importable factory path.
- Put calls behind a small module function or private method that tests can monkeypatch. Never require live HTTP, Elasticsearch, model downloads, or credentials in unit tests.
- Import an optional dependency inside `run()`, `warm_up()`, or a private loader and raise an actionable `ImportError` only when the component is used.

## Documents and serialization

When returning the same documents unchanged, returning the original objects is acceptable if the mutation policy says the component is read-only. When changing content, metadata, score, or embedding, create a new `Document` unless in-place mutation is an explicit contract:

```python
updated = Document(
    id=document.id,
    content=new_content,
    meta=dict(document.meta or {}),
    score=document.score,
    embedding=getattr(document, "embedding", None),
)
```

Preserve `id`, `content`, `meta`, `score`, and `embedding` unless the component intentionally changes or drops a field. Copy `meta` before modifying it. Define stable IDs and provenance metadata for newly created chunks.

For ordinary serializable constructor values, rely on the `@component` decorator's automatic persistence. Add custom `to_dict()`/`from_dict()` only when a value needs an explicit reversible representation:

```python
from haystack import default_from_dict, default_to_dict

def to_dict(self) -> dict[str, object]:
    return default_to_dict(self, factory_path=self.factory_path)

@classmethod
def from_dict(cls, data: dict[str, object]) -> "MyComponent":
    return default_from_dict(cls, data)
```

The serialized form must have this shape:

```yaml
type: some_package.some_module.MyComponent
init_parameters:
  factory_path: some_package.factories.build_client
```

Do not serialize live clients, open files, locks, locally defined functions, or loaded models. Store only the configuration needed to recreate them.

## Exports and import paths

- Put a shared component in the closest category module under `packages/retrieval-components/src/retrieval_components/`.
- Export it from that category's `__init__.py`. Add a top-level `retrieval_components` export when the package's existing convenient public surface should include it.
- Put a small project-owned component in the project's established `components.py` module unless the project already uses another layout.
- Use the defining module's fully qualified import path in YAML. Do not rely on a top-level re-export for deserialization.
- Ensure the owning package is installed or on the Python path of the process that calls `AsyncPipeline.loads()`.

## Component fragments

A component fragment contains Haystack serialization data, not a complete pipeline and not Hydra object-instantiation syntax:

```yaml
type: retrieval_components.ranking.embedding_similarity_ranker.EmbeddingSimilarityRanker
init_parameters:
  top_k: 10
  similarity: ${selections.embedding_model.similarity}
```

For a constructor with no required configuration, omit `init_parameters`:

```yaml
type: retrieval_components.interfaces.stage_io.InferenceInput
```

Rules:

- Use `type`, never `_target_`.
- Set `type` to `<importable.module>.<ClassName>`.
- Set `init_parameters` to constructor keyword arguments only; never put runtime socket inputs there.
- Use YAML `null` for Python `None`.
- Quote strings that YAML could coerce or that contain regex/backslash syntax. In double-quoted YAML regexes, escape backslashes, for example `"\\s+"`.
- Use `${path.to.value}` for OmegaConf interpolation.
- Use `???` for a required value or required config choice so composition fails early.
- Put shared semantic model selections under root `selections` and interpolate them into component fragments.

## Pipeline topology

After Hydra composition, the resolved `pipeline` subtree must be valid Haystack pipeline serialization with exactly these top-level concepts:

```yaml
components:
  input:
    type: retrieval_components.interfaces.stage_io.InferenceInput
  selector:
    type: retrieval_components.cascade.top_documents.TopDocuments
    init_parameters:
      top_k: 10
  output:
    type: retrieval_components.interfaces.stage_io.InferenceOutput

connections:
  - sender: input.candidate_documents
    receiver: selector.documents
  - sender: selector.documents
    receiver: output.documents

max_runs_per_component: 100
metadata:
  description: Select top candidate documents.
```

Connection endpoints use `<component_instance_name>.<socket_name>`. The component instance name is the key under `components`; it is not necessarily the Python class name.

Check every connection against the Python contract:

- The sender suffix must be a declared output socket.
- The receiver suffix must be a declared input socket.
- Types must be compatible.
- Every required non-entry input must have a sender.
- The repository's stage pipelines normally expose inputs and collect outputs through the `InferenceInput`, `InferenceOutput`, or `IndexingOutput` boundary components.

For dynamic fusion configured with `weights: {lexical: 1.0, dense: 2.0}`, the receiver sockets are `fusion.lexical` and `fusion.dense`; connect each source to the matching name.

## Hydra composition and placement

Select a reusable component fragment into a pipeline's `components` mapping with a defaults entry:

```yaml
defaults:
  - /component/ranker@components.ranker: embedding_similarity
  - _self_
```

Interpret this as:

- `/component/ranker` is the config group.
- `embedding_similarity` selects `component/ranker/embedding_similarity.yaml`.
- `@components.ranker` packages that fragment under `pipeline.components.ranker` when this topology is composed as `pipeline`.
- `_self_` controls where the current file participates in precedence. Keep it last when local topology values should override earlier defaults, following neighboring configs.

A typical topology combines selected fragments and inline boundary or project components:

```yaml
defaults:
  - /component/query_parser@components.query_parser: content_field
  - /component/retriever@components.retriever: jsonl_embeddings
  - _self_

components:
  input:
    type: retrieval_components.interfaces.stage_io.InferenceInput
  project_transform:
    type: my_project.components.ProjectTransform
    init_parameters:
      separator: " "
  output:
    type: retrieval_components.interfaces.stage_io.InferenceOutput

connections:
  - sender: input.query_meta
    receiver: query_parser.meta
  - sender: query_parser.text
    receiver: project_transform.query
  - sender: project_transform.query
    receiver: retriever.query
  - sender: retriever.documents
    receiver: output.documents

max_runs_per_component: 100
metadata:
  description: Project inference topology.
```

Placement:

- Cross-project fragment: `packages/retrieval-core/src/retrieval_core/configs/component/<group>/<choice>.yaml`.
- Cross-project topology: the matching core `configs/pipeline/<stage>/` group.
- Project-wide fragment or topology: `projects/<project>/configs/<group>/<package_name>/<choice>.yaml`, selected as `<package_name>/<choice>` within the unchanged group.
- One-experiment fragment or topology: that experiment's `configs/<group>/<experiment_namespace>/<choice>.yaml` overlay, selected as `<experiment_namespace>/<choice>`.
- Experiment base entrypoint: `experiments/<experiment>/configs/base-experiment-configs/`.
- Concrete run entrypoint: `experiments/<experiment>/configs/runs/`.

Keep shared/core choices unqualified. Namespace the choice, not the Hydra group:

```yaml
# Correct: replaces the core pipeline/inference choice with a project-owned choice.
- override /pipeline/inference@pipeline: my_project/my_topology

# Incorrect: creates a different group and leaves pipeline/inference unresolved.
- override /project-configs/pipeline/inference@pipeline: my_topology
```

Do not place reusable component fragments in `base-experiment-configs/` or `runs/`. A run may override constructor values after composition:

```yaml
pipeline:
  components:
    retriever:
      init_parameters:
        top_k: 100
```

## Tests and validation

Test the Python behavior directly without a pipeline:

```python
from haystack import Document

from retrieval_components.cascade import TopDocuments


def test_top_documents_keeps_highest_scores() -> None:
    component_instance = TopDocuments(top_k=1)
    result = component_instance.run(
        [Document(id="low", score=0.1), Document(id="high", score=0.9)]
    )
    assert [document.id for document in result["documents"]] == ["high"]
```

Cover constructor validation, each output, optional input behavior, empty input, stable ordering or ties when relevant, field preservation, and mocked external failures. For dynamic sockets, test at least two configured names.

Test configuration by composing a real consuming config and asking Haystack to load it:

```python
from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline, to_container


def test_pipeline_with_component_loads() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=cpu",
            "pipeline/inference@pipeline=my_topology",
        ],
    )
    pipeline_dict = to_container(cfg.pipeline)
    pipeline = load_async_pipeline(cfg.pipeline)

    assert "selector" in pipeline.graph.nodes
    assert pipeline_dict["components"]["selector"]["init_parameters"]["top_k"] == 10
```

`load_async_pipeline` resolves the OmegaConf subtree to plain containers, emits Haystack-shaped YAML, and calls `AsyncPipeline.loads()`. A mere YAML parse is insufficient: it does not verify imports, constructor parameters, sockets, or connection compatibility.

Run the focused owning-package test first, then config loading tests, then the full owning-package suite. Do not invoke a live model or service to prove unit configuration.

## Failure checklist

Before finishing, reject these common mistakes:

- Haystack 1.x API (`BaseComponent`, `@component.output_types` omitted, `Pipeline.add_node`, or `outgoing_edges`).
- Hydra `_target_` in a Haystack component entry.
- Runtime inputs incorrectly placed under `init_parameters`.
- `T | None` used without a default for an intended optional input socket.
- Returned keys that differ from declared output names.
- Dynamic socket setup with a `run()` method that lacks `**kwargs`.
- `component.set_output_types` combined with an output decorator.
- Non-serializable constructor state or heavy work in `__init__`.
- A YAML `type` that points only to an `__init__.py` re-export instead of the defining module.
- A connection name or type that does not match the component sockets.
- A fragment placed in a run entrypoint instead of the correct reusable config group.
- Unit tests that require downloads, credentials, network access, Elasticsearch, or other external state.
