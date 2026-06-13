"""Model component aliases that are already provided by Haystack.

These imports intentionally keep model-heavy implementations in Haystack. The
project can reference these classes directly in Hydra configs when the optional
runtime dependencies are installed.
"""

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.rankers import (
    SentenceTransformersSimilarityRanker,
    TransformersSimilarityRanker,
)

__all__ = [
    "SentenceTransformersDocumentEmbedder",
    "SentenceTransformersSimilarityRanker",
    "SentenceTransformersTextEmbedder",
    "TransformersSimilarityRanker",
]
