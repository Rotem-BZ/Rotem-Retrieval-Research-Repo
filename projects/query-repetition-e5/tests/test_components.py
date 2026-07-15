from query_repetition_e5.components import QueryRepeater


def test_query_repeater_repeats_the_query_with_configured_separator() -> None:
    repeater = QueryRepeater(separator=" | ")

    assert repeater.run("where is the evidence?") == {
        "query": "where is the evidence? | where is the evidence?"
    }
