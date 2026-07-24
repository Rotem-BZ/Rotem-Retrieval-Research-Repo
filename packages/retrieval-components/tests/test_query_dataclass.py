from dataclasses import FrozenInstanceError

import pytest

from retrieval_components import Query


def test_query_is_available_from_the_package_root_and_preserves_nested_meta() -> None:
    source_meta = {"filters": {"language": "en"}, "weights": [1, 2]}

    query = Query(id="q1", meta=source_meta)
    source_meta["filters"]["language"] = "fr"

    assert query.to_dict() == {
        "id": "q1",
        "content": None,
        "meta": {"filters": {"language": "en"}, "weights": [1, 2]},
    }
    assert Query.from_dict(query.to_dict()) == query


def test_with_content_returns_a_new_query_without_changing_the_source() -> None:
    source = Query(id="q1", meta={"question": "Where?"})

    parsed = source.with_content("Where?")

    assert source.content is None
    assert parsed.content == "Where?"
    assert parsed.id == source.id
    assert parsed.meta == source.meta


def test_query_fields_cannot_be_reassigned() -> None:
    query = Query(id="q1")

    with pytest.raises(FrozenInstanceError):
        query.content = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("content", [1, {}, []])
def test_query_rejects_non_string_content(content: object) -> None:
    with pytest.raises(TypeError, match="content"):
        Query(id="q1", content=content)  # type: ignore[arg-type]
