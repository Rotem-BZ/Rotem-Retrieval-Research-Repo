# retrieval-components

The publishable component library for the retrieval research monorepo. It contains
reusable Haystack components without experiment orchestration. Component categories
live directly under the `retrieval_components` package.

## Available components

The category modules below are the supported inventory. Each module exports the
listed classes from its `__init__.py`.

| Import | Available components | Purpose |
| --- | --- | --- |
| `retrieval_components.cascade` | `ChunkCascade`, `TopKDocuments`, `TopPDocuments` | Cap chunks per source document, select a fixed count, or select cumulative score mass from a ranked list. |
| `retrieval_components.chunking` | `LangChainDocumentSplitter` | Adapt a `langchain_text_splitters` splitter to Haystack documents. |
| `retrieval_components.filtering` | `DocumentContentFilter` | Filter documents by regex and word-count bounds. |
| `retrieval_components.fusion` | `LinearScoreFusion`, `ReciprocalRankFusion`, `ScoreFusion`, `ZScoreFusion` | Fuse dynamic named document inputs with source weights, with separate min-max and Z-normalized variants. |
| `retrieval_components.indexing` | `ElasticsearchDocumentIndexer`, `JsonlDocumentIndexer` | Write documents to Elasticsearch or a local JSONL artifact. |
| `retrieval_components.interfaces` | `IndexingInput`, `IndexingOutput`, `InferenceInput`, `InferenceOutput` | Define fixed stage-boundary sockets for indexing and inference. |
| `retrieval_components.models` | `SentenceTransformersDocumentEmbedder`, `SentenceTransformersSimilarityRanker`, `SentenceTransformersTextEmbedder`, `TransformersSimilarityRanker` | Re-export native Haystack model components for categorized imports. |
| `retrieval_components.preprocessing` | `DocumentContentFieldParser`, `DocumentTextPrefixer`, `QueryContentAdapter`, `QueryContentFieldParser`, `QueryTextPreprocessor`, `TextPreprocessor` | Materialize content from metadata fields, transform query values, and adapt them to native text sockets. |
| `retrieval_components.ranking` | `EmbeddingSimilarityRanker` | Rank already-embedded documents against a query embedding. |
| `retrieval_components.reformulation` | `HttpQueryReformulator` | Call an injected HTTP reformulation service. |
| `retrieval_components.retrieval` | `ElasticsearchBM25Retriever`, `JsonlEmbeddingRetriever`, `JsonlKeywordRetriever` | Retrieve from Elasticsearch or local JSONL artifacts. |

Most repo-defined classes are also available from `retrieval_components` for
convenience. Native model aliases remain under `retrieval_components.models`.
The shared inference value is available directly as
`from retrieval_components import Query`; it carries `id`, optional `content`, and
arbitrary nested metadata between query parsers and preprocessors. Treat it as an
immutable value and use `query.with_content(...)` in transformation components.

## Haystack overlap

This package prefers native Haystack components when they already satisfy the
required contract:

- The four classes in `retrieval_components.models` are direct Haystack re-exports.
- `DocumentTextPrefixer` and `TextPreprocessor` add prefix/suffix and small regex
  transforms beyond the relevant native cleaner contracts.
- `DocumentContentFieldParser` and `QueryContentFieldParser` provide strict dataset-field
  boundaries before experiment-specific metadata renderers run.
- The fusion components add weighted, dynamic named sockets beyond the fixed-input
  use cases covered by `DocumentJoiner`. `LinearScoreFusion` and `ZScoreFusion`
  provide distinct per-source normalization contracts.
- The JSONL components provide the repository's local artifact contract; the
  Elasticsearch components provide small injectable-client boundaries used by its
  pipelines.

`JsonlDocumentIndexer` also supports the indexing stage's incremental write session.
During that session, repeated `documents` inputs append to a stage-owned temporary
JSONL artifact; ordinary direct `run()` calls retain their one-shot behavior.

Optional runtime packages are imported only when the relevant component is used.
Tests mock HTTP, Elasticsearch, and LangChain integration points.
