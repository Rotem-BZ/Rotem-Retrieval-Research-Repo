"""Expose ordered project and retrieval-core config roots to Hydra."""

from __future__ import annotations

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from retrieval_core.utils.config.searchpath import active_config_fallbacks


class RetrievalCoreSearchPathPlugin(SearchPathPlugin):
    """Append configured fallbacks after Hydra's primary config directory."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        paths = active_config_fallbacks()
        if paths is None:
            paths = (("retrieval-core", "pkg://retrieval_core.configs"),)
        for provider, path in paths:
            search_path.append(provider=provider, path=path)
