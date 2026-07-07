"""Single-text preprocessing component."""

from __future__ import annotations

from haystack import component

from retrieval_research.utils.text import apply_text_transforms


@component
class TextPreprocessor:
    """Clean and optionally prefix/suffix a single text input."""

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        strip: bool = True,
        lowercase: bool = False,
        replace_regexes: dict[str, str] | None = None,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.strip = strip
        self.lowercase = lowercase
        self.replace_regexes = replace_regexes or {}

    @component.output_types(text=str)
    def run(self, text: str) -> dict[str, str]:
        return {
            "text": apply_text_transforms(
                text,
                prefix=self.prefix,
                suffix=self.suffix,
                strip=self.strip,
                lowercase=self.lowercase,
                replace_regexes=self.replace_regexes,
            )
        }
