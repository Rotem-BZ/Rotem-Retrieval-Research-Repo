from pathlib import Path

from retrieval_core.cli import (
    _extract_dry_run,
    _extract_modes,
    _run_stage_with_config,
    run_stage,
    usage,
    validate_stage,
)
from retrieval_core.stages import STAGE_RUNNERS
from retrieval_core.utils.io import read_json


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation", "prepare_mapping"}


def test_usage_lists_default_stages() -> None:
    help_text = usage()

    assert "Usage: stage" in help_text
    assert "<stage-name>" in help_text
    assert "materialized/production/toy_dense_indexing_reference" in help_text
    assert "--dry-run" in help_text
    assert "--validate" in help_text
    assert "build-command" in help_text
    assert "indexing" in help_text
    assert "inference" in help_text
    assert "evaluation" in help_text
    assert "prepare_mapping" in help_text


def test_extract_dry_run_removes_flag() -> None:
    assert _extract_dry_run(["--dry-run", "indexing", "dataset=toy"]) == (
        True,
        ["indexing", "dataset=toy"],
    )


def test_extract_modes_rejects_dry_run_with_validate() -> None:
    import pytest

    with pytest.raises(SystemExit, match="mutually exclusive"):
        _extract_modes(["--dry-run", "--validate", "indexing"])
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
    assert Path(result["output"]["index_path"]).name == "index.jsonl"
    assert "stage-dry-run-" in str(result["output"]["index_path"])
    assert not tmp_path.joinpath("artifacts").exists()


def test_validate_checks_pipeline_without_executing_or_writing(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()

    result = validate_stage(
        "indexing",
        [
            f"paths.project_root={tmp_path.as_posix()}",
            "dataset=toy",
            f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
            f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
            f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
            "pipeline/indexing@pipeline=dummy_jsonl",
        ],
    )

    assert result == {"valid": True, "stage": "indexing"}
    assert not tmp_path.joinpath("artifacts").exists()


def test_dry_run_inference_reads_real_index_but_saves_no_run(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()
    common_overrides = [
        f"paths.project_root={tmp_path.as_posix()}",
        "dataset=toy",
        f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
        f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
        f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
    ]
    run_stage(
        "indexing",
        [*common_overrides, "pipeline/indexing@pipeline=dummy_jsonl", "stage.run_name=toy"],
    )
    indexing_runs = list((tmp_path / "artifacts" / "runs" / "indexing").iterdir())
    assert len(indexing_runs) == 1
    manifest = read_json(indexing_runs[0] / "manifest.json")
    assert manifest["stage"]["run_id"] == indexing_runs[0].name
    assert Path(manifest["artifacts"]["index"]).parent == indexing_runs[0]

    predictions = run_stage(
        "inference",
        [
            *common_overrides,
            "pipeline/inference@pipeline=dummy_keyword",
            f"stage.indexing_run_id={indexing_runs[0].name}",
        ],
        dry_run=True,
    )

    assert len(predictions) == 3
    assert not (tmp_path / "artifacts" / "runs" / "inference").exists()


def test_prepare_mapping_stage_reuses_content_addressed_mapping(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()
    overrides = [
        f"paths.project_root={tmp_path.as_posix()}",
        "dataset=toy",
        f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
        f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
        f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
        "input_mapping=dev_tiny",
    ]

    first = run_stage("prepare_mapping", overrides)
    second = run_stage("prepare_mapping", overrides)

    assert first["reused"] is False
    assert second["reused"] is True
    assert first["mapping_path"] == second["mapping_path"]


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
