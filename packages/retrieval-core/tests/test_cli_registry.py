from pathlib import Path

import pytest
from haystack import Document, component

from retrieval_core.cli import main, run_stage
from retrieval_core.stages import STAGES
from retrieval_core.stages.base import Stage
from retrieval_core.utils.config import core_config_dir
from retrieval_core.utils.io import read_json
from retrieval_core.utils.logging import RUN_LOG_FILENAME


@component
class _DocumentPassThrough:
    def __init__(self, **_: object) -> None:
        pass

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": documents}


def test_stage_registry_contains_default_stages() -> None:
    assert set(STAGES) == {"indexing", "inference", "evaluation", "prepare_mapping"}
    assert all(issubclass(stage_type, Stage) for stage_type in STAGES.values())


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
    caplog,
    tmp_path: Path,
) -> None:
    marker = "FULL_RESULT_SHOULD_NOT_BE_PRINTED"
    caplog.set_level("INFO", logger="retrieval_core")

    class FakeIndexingStage(Stage):
        def prepare_config(self) -> None:
            pass

        def run(self):
            return [{"marker": marker}]

    monkeypatch.setitem(STAGES, "indexing", FakeIndexingStage)

    result = main(
        [
            "indexing",
            "dataset=toy",
            "pipeline/indexing@pipeline=dense/documents_in_memory",
            "selections/embedding_model=e5/small_v2",
            "runtime=cpu",
            "selections.index_id=test-index",
            f'paths.project_root="{tmp_path.as_posix()}"',
            "stage.run_id=logging-test",
        ]
    )

    output = capsys.readouterr().out
    assert result is None
    assert marker not in output
    assert marker not in caplog.text
    assert "'result_count': 1" in caplog.text
    log_path = tmp_path / "artifacts" / "runs" / "indexing" / "logging-test" / RUN_LOG_FILENAME
    assert log_path.is_file()
    assert "'result_count': 1" in log_path.read_text(encoding="utf-8")


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
    assert (
        mapping_path
        == tmp_path / "artifacts" / "input_mappings" / "toy_dev_tiny" / "input_mapping.json"
    )
    assert Path(result["metadata_path"]) == mapping_path.parent / "meta.json"


def test_materialized_config_dispatches_by_declared_stage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class FakeIndexingStage(Stage):
        def prepare_config(self) -> None:
            pass

        def prepare(self) -> None:
            self.prepare_config()
            self.output_dir = tmp_path

        def run(self):
            captured["run_id"] = self.cfg.stage.run_id
            captured["output_dir"] = self.cfg.stage.output_dir
            captured["preserve_run_config"] = self.cfg.stage.preserve_run_config
            return {"ok": True}

    monkeypatch.setitem(STAGES, "indexing", FakeIndexingStage)

    entrypoint = (
        core_config_dir() / "materialized" / "production" / "toy_dense_indexing_reference.yaml"
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
        "pipeline/indexing@pipeline=dense/documents_in_memory",
        "selections/embedding_model=e5/small_v2",
        "pipeline.components.embedder.type=test_cli_registry._DocumentPassThrough",
        "pipeline.components.embedder.init_parameters={}",
        "runtime=cpu",
        "runtime.progress_bar=false",
        "runtime.indexing_batch_size=2",
        "selections.index_id=toy-index",
    ]

    main(["indexing", *common_overrides, "stage.run_id=indexing-one"])

    index_path = tmp_path / "artifacts" / "indexes" / "toy-index" / "index.json"
    manifest_path = tmp_path / "artifacts" / "runs" / "indexing" / "indexing-one" / "manifest.json"
    assert index_path.is_file()
    manifest_inputs = read_json(manifest_path)["inputs"]
    assert manifest_inputs["index_id"] == "toy-index"
    assert manifest_inputs["dataset"] == "toy"
    assert manifest_inputs["documents_path"] == str(dataset_dir / "documents.jsonl")
    assert manifest_inputs["document_count"] == 4
    assert manifest_inputs["batch_count"] == 2
    assert len(manifest_inputs["documents_sha256"]) == 64

    with pytest.raises(FileExistsError, match="choose another selections.index_id"):
        run_stage(["indexing", *common_overrides, "stage.run_id=indexing-two"])
    assert not (tmp_path / "artifacts" / "runs" / "indexing" / "indexing-two").exists()


def test_failed_stage_records_traceback_in_run_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "stage exploded"

    class FailingIndexingStage(Stage):
        def prepare_config(self) -> None:
            pass

        def run(self):
            raise RuntimeError(marker)

    monkeypatch.setitem(STAGES, "indexing", FailingIndexingStage)
    overrides = [
        "dataset=toy",
        "pipeline/indexing@pipeline=dense/documents_in_memory",
        "selections/embedding_model=e5/small_v2",
        "runtime=cpu",
        "selections.index_id=failing-index",
        f'paths.project_root="{tmp_path.as_posix()}"',
        "stage.run_id=failing-run",
    ]

    with pytest.raises(RuntimeError, match=marker):
        run_stage(["indexing", *overrides])

    log_path = tmp_path / "artifacts" / "runs" / "indexing" / "failing-run" / RUN_LOG_FILENAME
    contents = log_path.read_text(encoding="utf-8")
    assert "Stage failed" in contents
    assert "RuntimeError: stage exploded" in contents
