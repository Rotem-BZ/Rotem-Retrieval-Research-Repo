"""Project-specific Haystack components."""

from haystack import component


@component
class {{ cookiecutter.component_class_name }}:
    """Apply the project-local treatment to a raw query.

    The generated implementation is intentionally an identity transform. Replace
    the body of ``run`` and update its focused unit test for the actual experiment.
    """

    @component.output_types(query=str)
    def run(self, query: str) -> dict[str, str]:
        return {"query": query}
