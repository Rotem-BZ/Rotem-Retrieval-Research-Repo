"""Expose retrieval-core's config groups to every consuming project."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class RetrievalCoreSearchPathPlugin(SearchPathPlugin):
    """Append core configs after the consuming project's primary config directory."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="retrieval-core", path="pkg://retrieval_core.configs")
