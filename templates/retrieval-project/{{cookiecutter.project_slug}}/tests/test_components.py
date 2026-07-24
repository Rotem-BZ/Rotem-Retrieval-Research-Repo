from {{ cookiecutter.package_name }}.components import {{ cookiecutter.component_class_name }}
from retrieval_components import Query


def test_query_transformer_returns_the_treatment_query() -> None:
    transformer = {{ cookiecutter.component_class_name }}()
    query = Query(id="q1", content="where is the evidence?")

    # The scaffold starts as an identity/parity treatment. Replace this expectation
    # when implementing the experiment-specific transformation.
    assert transformer.run(query) == {"query": query}
