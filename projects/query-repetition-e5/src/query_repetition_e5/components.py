"""Project-specific Haystack components."""

from haystack import component


@component
class QueryRepeater:
    """Repeat the original query before it is passed to the E5 preprocessor."""

    def __init__(self, separator: str = " ") -> None:
        self.separator = separator

    @component.output_types(query=str)
    def run(self, query: str) -> dict[str, str]:
        return {"query": self.separator.join((query, query))}
