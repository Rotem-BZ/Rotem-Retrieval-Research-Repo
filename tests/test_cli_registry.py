from pathlib import Path

from retrieval_research.cli import _extract_dry_run, run_stage, usage
from retrieval_research.stages import STAGE_RUNNERS


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation"}


def test_usage_lists_default_stages() -> None:
    help_text = usage()

    assert "Usage: rr" in help_text
    assert "<stage>" in help_text
    assert "--dry-run" in help_text
    assert "indexing" in help_text
    assert "inference" in help_text
    assert "evaluation" in help_text


def test_extract_dry_run_removes_flag() -> None:
    assert _extract_dry_run(["--dry-run", "indexing", "dataset=toy"]) == (
        True,
        ["indexing", "dataset=toy"],
    )
    assert _extract_dry_run(["indexing", "--dry-run", "dataset=toy"]) == (
        True,
        ["indexing", "dataset=toy"],
    )


def test_dry_run_redirects_index_artifacts(tmp_path: Path) -> None:
    documents_path = Path("data/processed/toy/documents.jsonl").resolve()

    result = run_stage(
        "indexing",
        [
            f"paths.project_root={tmp_path.as_posix()}",
            "dataset=toy",
            f'dataset.documents_path="{documents_path.as_posix()}"',
            "pipeline/indexing@pipeline=dummy_jsonl",
        ],
        dry_run=True,
    )

    assert result["indexer"]["indexed_count"] == 4
    assert not tmp_path.joinpath("artifacts").exists()
