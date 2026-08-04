# retrieval-components

The publishable component library for the retrieval research monorepo. It contains
reusable Haystack components without experiment orchestration. Component categories
live directly under the `retrieval_components` package.

## Available components

The defining modules below are the supported inventory. Import classes from these
full module paths; package initializers deliberately do not re-export them.

| Import | Available components | Purpose |
| --- | --- | --- |
| `retrieval_components.cascade.chunk_cascade` | `ChunkCascade` | Cap chunks per source document. |
| `retrieval_components.cascade.top_k_documents` | `TopKDocuments` | Select a fixed count from a ranked list. |
| `retrieval_components.chunking.langchain_document_splitter` | `LangChainDocumentSplitter` | Recursively split documents by character length or an optional Hugging Face tokenizer. |
| `retrieval_components.experimental.elasticsearch_bm25_retriever` | `ElasticsearchBM25Retriever` | Incubate an Elasticsearch retriever whose interface is not yet stable. |
| `retrieval_components.experimental.elasticsearch_document_indexer` | `ElasticsearchDocumentIndexer` | Incubate an Elasticsearch indexer whose interface is not yet stable. |
| `retrieval_components.filtering.document_content_filter` | `DocumentContentFilter` | Filter documents by regex and word-count bounds. |
| `retrieval_components.fusion.normalized_score_fusion` | `LinearScoreFusion`, `ZScoreFusion` | Fuse scores using min-max or Z-score normalization. |
| `retrieval_components.fusion.reciprocal_rank_fusion` | `ReciprocalRankFusion` | Fuse ranked lists using reciprocal ranks. |
| `retrieval_components.fusion.score_fusion` | `ScoreFusion` | Sum weighted document scores. |
| `retrieval_components.indexing.persisted_in_memory_document_indexer` | `PersistedInMemoryDocumentIndexer` | Persist batched documents in a Haystack in-memory store. |
| `retrieval_components.interfaces.indexing` | `IndexingInput`, `IndexingOutput` | Define fixed indexing-stage boundary sockets. |
| `retrieval_components.interfaces.inference` | `InferenceInput`, `InferenceOutput` | Define fixed inference-stage boundary sockets. |
| `retrieval_components.models.sentence_transformers_similarity_ranker` | `SentenceTransformersSimilarityRanker` | Rank documents with a Query-aware sentence-transformer model. |
| `retrieval_components.models.sentence_transformers_text_embedder` | `SentenceTransformersTextEmbedder` | Embed a materialized Query with a sentence-transformer model. |
| `retrieval_components.models.transformers_similarity_ranker` | `TransformersSimilarityRanker` | Rank documents with a Query-aware Transformers model. |
| `retrieval_components.preprocessing.document_text_prefixer` | `DocumentTextPrefixer` | Prefix, suffix, or clean materialized document content. |
| `retrieval_components.preprocessing.identity_parser` | `IdentityParser` | Validate already-materialized document content. |
| `retrieval_components.preprocessing.query_text_preprocessor` | `QueryTextPreprocessor` | Prefix, suffix, or clean materialized Query content. |
| `retrieval_components.preprocessing.query_to_string` | `QueryToString` | Adapt a materialized Query to a plain string. |
| `retrieval_components.ranking.embedding_similarity_ranker` | `EmbeddingSimilarityRanker` | Rank already-embedded documents against a query embedding. |
| `retrieval_components.reformulation.http_query_reformulator` | `HttpQueryReformulator` | Call an injected HTTP reformulation service. |
| `retrieval_components.retrieval.persisted_in_memory_embedding_retriever` | `PersistedInMemoryEmbeddingRetriever` | Load a persisted Haystack in-memory store and delegate embedding retrieval to Haystack. |

Every category `__init__.py` is docstring-only, and the package root contains only
lightweight metadata. Importing a package or category therefore does not import its
component implementations or optional integrations. This matches the full defining-module
paths required by serialized Haystack component types.
Query-aware model subclasses remain under `retrieval_components.models`.
The shared inference value is available as
`from retrieval_components.dataclasses.query import Query`; it carries `id`, optional
`content`, and arbitrary nested metadata between query-aware components. Treat it as an
immutable value and use `query.with_content(...)` in transformation components.

## Haystack overlap

This package prefers native Haystack components when they already satisfy the
required contract:

- Query-facing classes in `retrieval_components.models` subclass their native
  Haystack implementations and accept `Query`. Import native document embedders
  directly from Haystack.
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
dependencies normally, while lightweight package initializers prevent those imports
until that component's defining module is selected. Tests mock HTTP, Elasticsearch, and
LangChain integration points.
