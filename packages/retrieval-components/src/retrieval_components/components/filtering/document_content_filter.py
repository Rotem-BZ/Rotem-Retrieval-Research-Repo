"""Content-based document and chunk filtering component."""

from __future__ import annotations

import re

from haystack import Document, component


@component
class DocumentContentFilter:
    """Filter documents by content regexes and word-count bounds."""

    def __init__(
        self,
        include_regex: str | None = None,
        exclude_regex: str | None = None,
        min_words: int | None = None,
        max_words: int | None = None,
        regex_flags: int = 0,
    ) -> None:
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex
        self.min_words = min_words
        self.max_words = max_words
        self.regex_flags = regex_flags
        self._include_pattern = re.compile(include_regex, regex_flags) if include_regex else None
        self._exclude_pattern = re.compile(exclude_regex, regex_flags) if exclude_regex else None

    @component.output_types(documents=list[Document], rejected_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        kept: list[Document] = []
        rejected: list[Document] = []

        for document in documents:
            target = document.content or ""
            if self._accepts(target):
                kept.append(document)
            else:
                rejected.append(document)

        return {"documents": kept, "rejected_documents": rejected}

    def _accepts(self, content: str) -> bool:
        word_count = len(re.findall(r"\b\w+\b", content))

        if self.min_words is not None and word_count < self.min_words:
            return False
        if self.max_words is not None and word_count > self.max_words:
            return False
        if self._include_pattern and not self._include_pattern.search(content):
            return False
        if self._exclude_pattern and self._exclude_pattern.search(content):
            return False
        return True
