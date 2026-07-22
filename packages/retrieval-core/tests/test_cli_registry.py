from pathlib import Path

import pytest

from retrieval_core.cli import main, run_stage
from retrieval_core.stages import STAGE_RUNNERS
from retrieval_core.utils.config import core_config_dir
from retrieval_core.utils.io import read_json


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGE_RUNNERS) == {"indexing", "inference", "evaluation", "prepare_mapping"}


def test_help_lists_default_stages(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out

    assert "usage: stage" in help_text
    assert "STAGE" in help_text
    assert "--entrypoint" in help_text
    assert "indexing" in help_text
    assert "inference" in help_text
    assert "evaluation" in help_text
    assert "prepare_mapping" in help_text
    assert "--experiment-dir" not in help_text
    assert "--config-dir" not in help_text


def test_console_main_does_not_return_or_print_full_stage_result(
    monkeypatch,
    capsys,
) -> None:
    marker = "FULL_RESULT_SHOULD_NOT_BE_PRINTED"
    monkeypatch.setitem(STAGE_RUNNERS, "indexing", lambda cfg: [{"marker": marker}])

    result = main(
        [
            "indexing",
            "dataset=toy",
            "pipeline/indexing@pipeline=dummy_jsonl",
            "runtime=cpu",
            "selections.index_id=test-index",
        ]
    )

    output = capsys.readouterr().out
    assert result is None
    assert marker not in output
    assert "'result_count': 1" in output


def test_prepare_mapping_stage_writes_run_id_mapping_directory(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()
    overrides = [
        f'paths.project_root="{tmp_path.as_posix()}"',
        "dataset=toy",
        f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
        f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
        f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
        "input_mapping_recipe=dev_tiny",
        "stage.run_id=toy_dev_tiny",
    ]

    result = run_stage(["prepare_mapping", *overrides])

    mapping_path = Path(result["mapping_path"])
    assert mapping_path == tmp_path / "artifacts" / "input_mappings" / "toy_dev_tiny" / "input_mapping.json"
    assert Path(result["metadata_path"]) == mapping_path.parent / "meta.json"


def test_materialized_config_dispatches_by_declared_stage(monkeypatch) -> None:
    captured = {}

    def fake_indexing_runner(cfg):
        captured["run_id"] = cfg.stage.run_id
        captured["output_dir"] = cfg.stage.output_dir
        captured["preserve_run_config"] = cfg.stage.preserve_run_config
        return {"ok": True}

    monkeypatch.setitem(STAGE_RUNNERS, "indexing", fake_indexing_runner)

    entrypoint = (
        core_config_dir()
        / "materialized"
        / "production"
        / "toy_dense_indexing_reference.yaml"
    )
    result = run_stage(["indexing", "--entrypoint", str(entrypoint)])

    assert result == {"ok": True}
    assert captured == {
        "run_id": "20260705_231537",
        "output_dir": "./artifacts/runs/indexing/20260705_231537",
        "preserve_run_config": True,
    }


def test_entrypoint_stage_must_match_requested_stage(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    entrypoint = configs / "evaluation.yaml"
    entrypoint.write_text(
        "defaults:\n  - /stages/evaluation\n  - override /dataset: toy\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        main(["inference", "--entrypoint", str(entrypoint)])


def test_indexing_publishes_an_immutable_selected_index(tmp_path: Path) -> None:
    dataset_dir = Path("data/processed/toy").resolve()
    common_overrides = [
        f'paths.project_root="{tmp_path.as_posix()}"',
        "dataset=toy",
        f'dataset.documents_path="{(dataset_dir / "documents.jsonl").as_posix()}"',
        f'dataset.queries_path="{(dataset_dir / "queries.jsonl").as_posix()}"',
        f'dataset.qrels_path="{(dataset_dir / "qrels.jsonl").as_posix()}"',
        "pipeline/indexing@pipeline=dummy_jsonl",
        "runtime=cpu",
        "runtime.progress_bar=false",
        "selections.index_id=toy-index",
    ]

    main(["indexing", *common_overrides, "stage.run_id=indexing-one"])

    index_path = tmp_path / "artifacts" / "indexes" / "toy-index" / "index.jsonl"
    manifest_path = (
        tmp_path / "artifacts" / "runs" / "indexing" / "indexing-one" / "manifest.json"
    )
    assert index_path.is_file()
    assert read_json(manifest_path)["inputs"]["index_id"] == "toy-index"

    with pytest.raises(FileExistsError, match="choose another selections.index_id"):
        main(["indexing", *common_overrides, "stage.run_id=indexing-two"])
    assert not (
        tmp_path / "artifacts" / "runs" / "indexing" / "indexing-two"
    ).exists()
