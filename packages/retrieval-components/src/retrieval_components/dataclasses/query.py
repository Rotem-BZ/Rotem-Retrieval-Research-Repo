"""Query value passed between inference components."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Query:
    """A query identifier, optional materialized content, and arbitrary metadata."""

    id: str
    content: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Query id must be a non-empty string.")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("Query content must be a string or None.")
        if not isinstance(self.meta, dict):
            raise TypeError("Query meta must be a dictionary.")
        object.__setattr__(self, "meta", deepcopy(self.meta))

    def with_content(self, content: str) -> Query:
        """Return a copy with materialized content."""

        return replace(self, content=content)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this query without exposing its mutable metadata."""

        return {
            "id": self.id,
            "content": self.content,
            "meta": deepcopy(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Query:
        """Construct a query from its serialized representation."""

        return cls(
            id=data["id"],
            content=data.get("content"),
            meta=data.get("meta") or {},
        )
