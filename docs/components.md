# Components

This repo prefers native Haystack components when they already cover the
contract, and adds project components only for missing glue or research-specific
behavior. The component package supports Haystack 2.30 and later 2.x releases.

| Need | Haystack status | Repo component |
| --- | --- | --- |
| Bi-encoder document/text models | Exists: `SentenceTransformersDocumentEmbedder`, `SentenceTransformersTextEmbedder`, retrievers such as `InMemoryEmbeddingRetriever` and `TextEmbeddingRetriever` | `retrieval_components.components.models` re-exports the Haystack classes for categorized imports |
| Cross encoder ranking | Exists: `SentenceTransformersSimilarityRanker`, `TransformersSimilarityRanker` | `retrieval_components.components.models` re-exports the Haystack classes |
| Document character cleanup | Exists: `DocumentCleaner`, `TextCleaner` | `DocumentTextPrefixer` adds missing prefix/suffix support |
| Query character cleanup | Exists: `TextCleaner` | `TextPreprocessor` adds prefix/suffix plus small regex cleanup |
| Document chunking | Exists: `DocumentSplitter`, `RecursiveDocumentSplitter` | `LangChainDocumentSplitter` adapts optional `langchain_text_splitters` splitters |
| Regex and word-count filtering | No direct single component found | `DocumentContentFilter` |
| HTTP query reformulation | No direct generic component found | `HttpQueryReformulator` |
| Elasticsearch indexing/querying | Not available in the local Haystack install without extra integrations | `ElasticsearchDocumentIndexer`, `ElasticsearchBM25Retriever` with injectable clients |
| RRF fusion | Exists: `DocumentJoiner(join_mode="reciprocal_rank_fusion")` | `ReciprocalRankFusion` supports dynamic named sockets and source weights |
| Score fusion | Partly exists through `DocumentJoiner` merge/distribution modes | `ScoreFusion` supports dynamic named sockets and weighted score sums |
| Cascade top-k/top-p | No direct cascade selectors found | `TopKDocuments`, `TopPDocuments` |

Optional runtime packages are imported only when the relevant component is used.
The test suite uses mocks for HTTP, Elasticsearch, and LangChain splitter
integration points.
