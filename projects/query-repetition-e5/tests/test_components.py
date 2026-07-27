from query_repetition_e5.components import QueryRepeater
from retrieval_components.dataclasses import Query


def test_query_repeater_repeats_the_query_with_configured_separator() -> None:
    repeater = QueryRepeater(separator=" | ")
    source = Query(id="q1", content="where is the evidence?", meta={"language": "en"})

    result = repeater.run(source)["query"]

    assert result.content == "where is the evidence? | where is the evidence?"
    assert result.meta == {"language": "en"}
    assert source.content == "where is the evidence?"
