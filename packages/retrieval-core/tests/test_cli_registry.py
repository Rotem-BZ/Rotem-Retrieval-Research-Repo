from pathlib import Path

import pytest

from retrieval_core.cli import main
from retrieval_core.stages import STAGE_RUNNERS


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation", "prepare_mapping"}


def test_help_lists_default_stages(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out

    assert "usage: stage" in help_text
    assert "STAGE_OR_CONFIG" in help_text
    assert "materialized/production/toy_dense_indexing_reference" in help_text
    assert "indexing" in help_text
    assert "inference" in help_text
    assert "evaluation" in help_text
    assert "prepare_mapping" in help_text

def test_prepare_mapping_stage_reuses_content_addressed_mapping(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()
    overrides = [
        f'paths.project_root="{tmp_path.as_posix()}"',
        "dataset=toy",
        f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
        f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
        f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
        "input_mapping=dev_tiny",
    ]

    first = main(["prepare_mapping", *overrides])
    second = main(["prepare_mapping", *overrides])

    assert first["reused"] is False
    assert second["reused"] is True
    assert first["mapping_path"] == second["mapping_path"]


def test_materialized_config_dispatches_by_declared_stage(monkeypatch) -> None:
    captured = {}

    def fake_indexing_runner(cfg):
        captured["run_id"] = cfg.stage.run_id
        captured["output_dir"] = cfg.stage.output_dir
        captured["preserve_run_config"] = cfg.stage.preserve_run_config
        return {"ok": True}

    monkeypatch.setitem(STAGE_RUNNERS, "indexing", fake_indexing_runner)

    result = main(["materialized/production/toy_dense_indexing_reference"])

    assert result == {"ok": True}
    assert captured == {
        "run_id": "20260705_231537",
        "output_dir": "./artifacts/runs/indexing/20260705_231537",
        "preserve_run_config": True,
    }
