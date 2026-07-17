from {{ cookiecutter.package_name }}.components import {{ cookiecutter.component_class_name }}


def test_query_transformer_returns_the_treatment_query() -> None:
    transformer = {{ cookiecutter.component_class_name }}()

    # The scaffold starts as an identity/parity treatment. Replace this expectation
    # when implementing the experiment-specific transformation.
    assert transformer.run("where is the evidence?") == {"query": "where is the evidence?"}
