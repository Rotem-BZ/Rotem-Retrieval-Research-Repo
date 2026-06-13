"""Small text preprocessing components not covered by Haystack cleaners."""

from __future__ import annotations

import re

from haystack import Document, component


def _apply_text_transforms(
    text: str,
    *,
    prefix: str,
    suffix: str,
    strip: bool,
    lowercase: bool,
    replace_regexes: dict[str, str],
) -> str:
    transformed = text
    for pattern, replacement in replace_regexes.items():
        transformed = re.sub(pattern, replacement, transformed)
    if strip:
        transformed = transformed.strip()
    if lowercase:
        transformed = transformed.lower()
    return f"{prefix}{transformed}{suffix}"


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
            "text": _apply_text_transforms(
                text,
                prefix=self.prefix,
                suffix=self.suffix,
                strip=self.strip,
                lowercase=self.lowercase,
                replace_regexes=self.replace_regexes,
            )
        }


@component
class DocumentTextPrefixer:
    """Apply the same text transform to each document's content."""

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

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        processed: list[Document] = []

        for document in documents:
            processed.append(
                Document(
                    id=document.id,
                    content=_apply_text_transforms(
                        document.content or "",
                        prefix=self.prefix,
                        suffix=self.suffix,
                        strip=self.strip,
                        lowercase=self.lowercase,
                        replace_regexes=self.replace_regexes,
                    ),
                    meta=dict(document.meta or {}),
                    score=getattr(document, "score", None),
                    embedding=getattr(document, "embedding", None),
                )
            )

        return {"documents": processed}
