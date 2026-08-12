"""Hydra and Haystack utilities for retrieval research experiments."""

from retrieval_core.data_schema import Qrel, document_from_dict, query_from_dict

__version__ = "0.1.0"

__all__ = ["Qrel", "__version__", "document_from_dict", "query_from_dict"]
