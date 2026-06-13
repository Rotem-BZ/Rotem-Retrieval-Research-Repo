"""Compatibility exports for Elasticsearch components."""

from retrieval_research.components.retrieval.elasticsearch_bm25_retriever import (
    ElasticsearchBM25Retriever,
)
from retrieval_research.components.retrieval.elasticsearch_document_indexer import (
    ElasticsearchDocumentIndexer,
)

__all__ = [
    "ElasticsearchBM25Retriever",
    "ElasticsearchDocumentIndexer",
]
