"""Query-aware model components and native document embedders.

Query-facing classes subclass their native Haystack implementations so the
repository Query value remains intact until the model boundary.
"""

from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from retrieval_components.models.sentence_transformers_similarity_ranker import (
    SentenceTransformersSimilarityRanker,
)
from retrieval_components.models.sentence_transformers_text_embedder import (
    SentenceTransformersTextEmbedder,
)
from retrieval_components.models.transformers_similarity_ranker import (
    TransformersSimilarityRanker,
)

__all__ = [
    "SentenceTransformersDocumentEmbedder",
    "SentenceTransformersSimilarityRanker",
    "SentenceTransformersTextEmbedder",
    "TransformersSimilarityRanker",
]
