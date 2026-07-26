"""Document text prefixing and cleanup component."""

from __future__ import annotations

from haystack import Document, component

from retrieval_components.preprocessing._text_transforms import apply_text_transforms


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
            if document.content is None:
                raise ValueError(
                    f"DocumentTextPrefixer requires document {document.id!r} to have content."
                )
            processed.append(
                Document(
                    id=document.id,
                    content=apply_text_transforms(
                        document.content,
                        prefix=self.prefix,
                        suffix=self.suffix,
                        strip=self.strip,
                        lowercase=self.lowercase,
                        replace_regexes=self.replace_regexes,
                    ),
                    meta=dict(document.meta),
                    score=document.score,
                    embedding=document.embedding,
                )
            )

        return {"documents": processed}
