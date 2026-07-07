from pathlib import Path

from retrieval_research.cli import _extract_dry_run, _run_stage_with_config, run_stage, usage
from retrieval_research.stages import STAGE_RUNNERS


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation"}


def test_usage_lists_default_stages() -> None:
    help_text = usage()

    assert "Usage: stage" in help_text
    assert "<stage-name>" in help_text
    assert "materialized/production/toy_dense_indexing_reference" in help_text
    assert "--dry-run" in help_text
    assert "build-command" in help_text
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
            "stage.run_name=toy_index",
        ],
        dry_run=True,
    )

    assert result["output"]["indexed_count"] == 4
    assert Path(result["output"]["index_path"]).name == "toy_index.jsonl"
    assert Path(result["output"]["index_path"]).parent.name == "indexes"
    assert not tmp_path.joinpath("artifacts").exists()


def test_materialized_config_dispatches_by_declared_stage(monkeypatch) -> None:
    captured = {}

    def fake_indexing_runner(cfg):
        captured["run_id"] = cfg.stage.run_id
        captured["output_dir"] = cfg.stage.output_dir
        return {"ok": True}

    monkeypatch.setitem(STAGE_RUNNERS, "indexing", fake_indexing_runner)

    stage_name, cfg, result = _run_stage_with_config(
        "materialized/production/toy_dense_indexing_reference"
    )

    assert stage_name == "indexing"
    assert result == {"ok": True}
    assert cfg.stage.preserve_run_config is True
    assert captured == {
        "run_id": "20260705_231537",
        "output_dir": "./artifacts/runs/indexing/20260705_231537",
    }
