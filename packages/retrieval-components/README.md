# retrieval-components

The publishable component library for the retrieval research monorepo. It contains
reusable Haystack components without experiment orchestration. Component categories
live directly under the `retrieval_components` package.

## Available components

The category modules below are the supported inventory. Each module exports the
listed classes from its `__init__.py`.

| Import | Available components | Purpose |
| --- | --- | --- |
| `retrieval_components.cascade` | `ChunkCascade`, `TopKDocuments` | Cap chunks per source document or select a fixed count from a ranked list. |
| `retrieval_components.chunking` | `LangChainDocumentSplitter` | Recursively split documents by character length or an optional Hugging Face tokenizer. |
| `retrieval_components.experimental` | `ElasticsearchBM25Retriever`, `ElasticsearchDocumentIndexer` | Incubate Elasticsearch components whose interfaces and behavior are not yet stable. |
| `retrieval_components.filtering` | `DocumentContentFilter` | Filter documents by regex and word-count bounds. |
| `retrieval_components.fusion` | `LinearScoreFusion`, `ReciprocalRankFusion`, `ScoreFusion`, `ZScoreFusion` | Fuse dynamic named document inputs with source weights, with separate min-max and Z-normalized variants. |
| `retrieval_components.indexing` | `PersistedInMemoryDocumentIndexer` | Persist batched documents in a Haystack in-memory store. |
| `retrieval_components.interfaces` | `IndexingInput`, `IndexingOutput`, `InferenceInput`, `InferenceOutput` | Define fixed stage-boundary sockets for indexing and inference. |
| `retrieval_components.models` | `SentenceTransformersDocumentEmbedder`, `SentenceTransformersSimilarityRanker`, `SentenceTransformersTextEmbedder`, `TransformersSimilarityRanker` | Provide Query-aware subclasses of native query model components and re-export the native document embedder. |
| `retrieval_components.preprocessing` | `DocumentTextPrefixer`, `IdentityParser`, `QueryTextPreprocessor`, `QueryToString` | Validate or transform materialized document and Query content, with an explicit compatibility boundary for plain-text components. |
| `retrieval_components.ranking` | `EmbeddingSimilarityRanker` | Rank already-embedded documents against a query embedding. |
| `retrieval_components.reformulation` | `HttpQueryReformulator` | Call an injected HTTP reformulation service. |
| `retrieval_components.retrieval` | `PersistedInMemoryEmbeddingRetriever` | Load a persisted Haystack in-memory store and delegate embedding retrieval to Haystack. |

Import public classes from their category packages. The package root stays lightweight
so Haystack can import only the defining module named by a serialized component type.
Query-aware model subclasses remain under `retrieval_components.models`.
The shared inference value is available as
`from retrieval_components.dataclasses import Query`; it carries `id`, optional `content`, and
arbitrary nested metadata between query-aware components. Treat it as an
immutable value and use `query.with_content(...)` in transformation components.

## Haystack overlap

This package prefers native Haystack components when they already satisfy the
required contract:

- Query-facing classes in `retrieval_components.models` subclass their native
  Haystack implementations and accept `Query`; `SentenceTransformersDocumentEmbedder`
  remains a direct re-export.
- `DocumentTextPrefixer` and `QueryTextPreprocessor` add prefix/suffix and small
  regex transforms beyond the relevant native cleaner contracts.
- `QueryToString` is retained for regular Haystack or project-owned components that
  still require a plain text socket.
- `IdentityParser` is the minimal document-parser implementation: it verifies that
  content was materialized at the dataset boundary and otherwise passes documents
  through unchanged. The parser slot remains available for richer future parsers.
- The fusion components add weighted, dynamic named sockets beyond the fixed-input
  use cases covered by `DocumentJoiner`. `LinearScoreFusion` and `ZScoreFusion`
  provide distinct per-source normalization contracts.
- The persisted in-memory components add the repository's atomic multi-batch artifact
  lifecycle and candidate-ID interface around Haystack's native document store and
  embedding retriever; similarity search remains Haystack-owned.
- Components under `retrieval_components.experimental` are deliberately excluded
  from the package-root API while their interfaces mature. The Elasticsearch
  components use Haystack's lazy-import boundary and initialize clients during
  pipeline warm-up.

The core indexing stage opens one `write_session()` on the reserved `indexer` component
and sends ordinary document batches through the pipeline. Successful session exit
atomically publishes the store; exceptional exit discards it.

Optional integrations such as Elasticsearch are imported only when the relevant
component is used. The chunking module imports its declared LangChain and Transformers
dependencies normally, while the lightweight package root prevents those imports until
that component's defining module is selected. Tests mock HTTP, Elasticsearch, and
LangChain integration points.
