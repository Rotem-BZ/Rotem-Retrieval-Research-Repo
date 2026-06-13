"""Custom Haystack components shared by research and production code."""

from retrieval_research.components.cascade import TopKDocuments, TopPDocuments
from retrieval_research.components.chunking import LangChainDocumentSplitter
from retrieval_research.components.dummy import (
    DummyDocumentSource,
    JsonlDocumentIndexer,
    JsonlDocumentSource,
    JsonlKeywordRetriever,
    WeightedDocumentFusion,
)
from retrieval_research.components.filtering import DocumentContentFilter
from retrieval_research.components.fusion import ReciprocalRankFusion, ScoreFusion
from retrieval_research.components.preprocessing import DocumentTextPrefixer, TextPreprocessor
from retrieval_research.components.reformulation import HttpQueryReformulator
from retrieval_research.components.retrieval import (
    ElasticsearchBM25Retriever,
    ElasticsearchDocumentIndexer,
    JsonlEmbeddingRetriever,
)

__all__ = [
    "DocumentContentFilter",
    "DocumentTextPrefixer",
    "DummyDocumentSource",
    "ElasticsearchBM25Retriever",
    "ElasticsearchDocumentIndexer",
    "HttpQueryReformulator",
    "JsonlDocumentIndexer",
    "JsonlDocumentSource",
    "JsonlEmbeddingRetriever",
    "JsonlKeywordRetriever",
    "LangChainDocumentSplitter",
    "ReciprocalRankFusion",
    "ScoreFusion",
    "TextPreprocessor",
    "TopKDocuments",
    "TopPDocuments",
    "WeightedDocumentFusion",
]
