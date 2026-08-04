"""Project-specific Haystack components."""

from haystack import component
from retrieval_components.dataclasses.query import Query


@component
class {{ cookiecutter.component_class_name }}:
    """Apply the project-local treatment to a raw query.

    The generated implementation is intentionally an identity transform. Replace
    the body of ``run`` and update its focused unit test for the actual experiment.
    """

    @component.output_types(query=Query)
    def run(self, query: Query) -> dict[str, Query]:
        return {"query": query}
