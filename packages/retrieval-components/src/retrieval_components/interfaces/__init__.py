"""Pipeline boundary components."""

from retrieval_components.interfaces.indexing import IndexingInput, IndexingOutput
from retrieval_components.interfaces.inference import InferenceInput, InferenceOutput

__all__ = [
    "IndexingInput",
    "IndexingOutput",
    "InferenceInput",
    "InferenceOutput",
]
