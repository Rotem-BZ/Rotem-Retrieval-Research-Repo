from retrieval_research.cli import usage
from retrieval_research.stages import STAGE_RUNNERS


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation"}


def test_usage_lists_default_stages() -> None:
    help_text = usage()

    assert "rr <stage>" in help_text
    assert "indexing" in help_text
    assert "inference" in help_text
    assert "evaluation" in help_text
