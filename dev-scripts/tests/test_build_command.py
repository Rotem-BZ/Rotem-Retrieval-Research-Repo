from pathlib import Path

import pytest

import build_command
from build_command import (
    HydraOverride,
    collect_selected_configs,
    discover_config_choices,
    editable_fields,
    effective_editable_fields,
    extract_required_defaults,
    find_active_config_dir,
    render_command,
    run_configure,
)


CONFIG_DIR = (
    Path(__file__).parents[2]
    / "packages"
    / "retrieval-core"
    / "src"
    / "retrieval_core"
    / "configs"
)
STAGES_CONFIG_DIR = CONFIG_DIR / "stages"


def test_main_exits_cleanly_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(build_command, "run_configure", interrupt)

    with pytest.raises(SystemExit) as exc_info:
        build_command.main([])

    assert exc_info.value.code == 130
    assert capsys.readouterr().out == "\nCommand builder cancelled.\n"


def test_finds_project_configs_from_nested_working_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "notebooks" / "analysis"
    configs = project / "configs"
    nested.mkdir(parents=True)
    configs.mkdir()

    assert find_active_config_dir(working_dir=nested) == configs.resolve()


def test_finds_experiment_configs_before_project_configs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    experiment = project / "experiments" / "example"
    nested = experiment / "notebooks"
    project_choice = project / "configs" / "dataset" / "project.yaml"
    experiment_choice = experiment / "configs" / "dataset" / "experiment.yaml"
    project_choice.parent.mkdir(parents=True)
    experiment_choice.parent.mkdir(parents=True)
    nested.mkdir()
    project_choice.write_text("name: project\n", encoding="utf-8")
    experiment_choice.write_text("name: experiment\n", encoding="utf-8")

    config_dir = find_active_config_dir(working_dir=nested)
    choices = discover_config_choices("dataset", config_dir=config_dir)

    assert config_dir == (experiment / "configs").resolve()
    assert [choice.name for choice in choices[:2]] == ["experiment", "project"]


def test_discovers_recursive_config_choices() -> None:
    choices = discover_config_choices(
        "selections/embedding_model", config_dir=CONFIG_DIR
    )

    assert [choice.name for choice in choices] == [
        "e5/base_v2",
        "e5/large_v2",
        "e5/small_v2",
    ]


def test_discovers_reranker_model_choices() -> None:
    choices = discover_config_choices(
        "selections/reranker_model", config_dir=CONFIG_DIR
    )

    assert [choice.name for choice in choices] == ["bge/v2_m3"]


def test_dataset_choices_do_not_include_nested_mapping_configs() -> None:
    choices = discover_config_choices("dataset", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == [
        "beir_scifact",
        "beir_scifact_smoke",
        "toy",
    ]


def test_discovers_builtin_input_mapping_recipe_choices() -> None:
    choices = discover_config_choices("input_mapping_recipe", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == [
        "dev_tiny",
        "judged_only",
        "random_smoke",
    ]


def test_extracts_required_defaults_from_stage_and_pipeline_configs() -> None:
    stage_required = extract_required_defaults(STAGES_CONFIG_DIR / "indexing.yaml")
    pipeline_required = extract_required_defaults(
        CONFIG_DIR / "pipeline" / "indexing" / "dense_jsonl.yaml"
    )

    assert [(item.group, item.override_key) for item in stage_required] == [
        ("dataset", "dataset"),
        ("pipeline/indexing", "pipeline/indexing@pipeline"),
        ("runtime", "runtime"),
    ]
    assert [(item.group, item.override_key) for item in pipeline_required] == [
        ("selections/embedding_model", "selections/embedding_model"),
    ]


def test_collects_selected_config_graph_with_mount_prefixes() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=STAGES_CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("runtime=gpu"),
        ],
        config_dir=CONFIG_DIR,
    )

    assert [(item.label, item.field_prefix, item.status) for item in selected] == [
        ("stage inference", "", "stage"),
        ("paths=local", "paths", "default"),
        ("dataset=toy", "dataset", "selected"),
        ("pipeline/inference@pipeline=dense_jsonl", "pipeline", "selected"),
        (
            "selections=index",
            "selections",
            "default",
        ),
        (
            "selections/embedding_model=e5/small_v2",
            "selections.embedding_model",
            "selected",
        ),
        (
            "component/query_preprocessor@pipeline.components.query_preprocessor=prefix_cleanup",
            "pipeline.components.query_preprocessor",
            "default",
        ),
        (
            "component/query_embedder@pipeline.components.query_embedder=sentence_transformers",
            "pipeline.components.query_embedder",
            "default",
        ),
        (
            "component/retriever@pipeline.components.retriever=jsonl_embeddings",
            "pipeline.components.retriever",
            "default",
        ),
        ("runtime=gpu", "runtime", "selected"),
    ]


def test_editable_fields_render_against_mounted_package_path() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=STAGES_CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("runtime=gpu"),
        ],
        config_dir=CONFIG_DIR,
    )
    embedding_model = selected[5]

    fields = editable_fields(embedding_model)

    assert (
        fields[-1].path
        == "selections.embedding_model.tokenizer_kwargs.model_max_length"
    )
    assert fields[-1].value == 512


def test_effective_editable_fields_reflect_current_overrides() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=STAGES_CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("selections.embedding_model.artifact_name=my_e5"),
            HydraOverride("runtime=gpu"),
        ],
        config_dir=CONFIG_DIR,
    )
    embedding_model = selected[5]

    fields = effective_editable_fields(
        embedding_model,
        stage_name="inference",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("selections.embedding_model.artifact_name=my_e5"),
            HydraOverride("runtime=gpu"),
        ],
    )

    artifact_name = next(
        field
        for field in fields
        if field.path == "selections.embedding_model.artifact_name"
    )
    assert artifact_name.value == "my_e5"


def test_effective_editable_fields_convert_composed_list_values() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=STAGES_CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("runtime=gpu"),
        ],
        config_dir=CONFIG_DIR,
    )
    pipeline = selected[3]

    fields = effective_editable_fields(
        pipeline,
        stage_name="inference",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("runtime=gpu"),
        ],
    )

    connections = next(
        field for field in fields if field.path == "pipeline.connections"
    )
    assert isinstance(connections.value, list)
    assert connections.value[0] == {
        "sender": "input.query",
        "receiver": "query_preprocessor.text",
    }


def test_render_command_preserves_hydra_override_syntax() -> None:
    command = render_command(
        "inference",
        [
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("runtime=gpu"),
        ],
    )

    assert (
        command == "uv run stage inference dataset=toy "
        "pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 "
        "runtime=gpu"
    )


def test_configure_flow_builds_indexing_dummy_command() -> None:
    result = _run_with_answers(["2", "3", "3", "2", "n", "toy-index", ""])

    assert result.command == (
        "uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl "
        "runtime=gpu selections.index_id=toy-index"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/indexing@pipeline=dummy_jsonl",
        "runtime=gpu",
        "selections.index_id=toy-index",
    )


def test_configure_flow_accepts_generated_dense_index_id(tmp_path: Path) -> None:
    result, output = _run_with_answers(
        ["2", "3", "2", "2", "3", "n", "", ""],
        include_output=True,
        indexes_dir=tmp_path / "indexes",
    )

    assert result.command == (
        "uv run stage indexing dataset=toy pipeline/indexing@pipeline=dense_jsonl "
        "runtime=gpu selections/embedding_model=e5/small_v2 "
        "selections.index_id=toy-e5-small-v2-index"
    )
    assert result.overrides[-1] == "selections.index_id=toy-e5-small-v2-index"
    assert "new index id [toy-e5-small-v2-index]: " in output


def test_configure_flow_builds_inference_dense_command_with_top_k(
    tmp_path: Path,
) -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "2",
            "3",
            "n",
            "1",
            "pipeline.components.retriever.init_parameters.top_k=100",
            "",
        ],
        indexes_dir=tmp_path / "indexes",
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "runtime=gpu "
        "selections/embedding_model=e5/small_v2 "
        "selections.index_id=index-1 "
        "pipeline.components.retriever.init_parameters.top_k=100"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "runtime=gpu",
        "selections/embedding_model=e5/small_v2",
        "selections.index_id=index-1",
        "pipeline.components.retriever.init_parameters.top_k=100",
    )


def test_configure_flow_builds_evaluation_command_with_metrics() -> None:
    result = _run_with_answers(
        ["1", "3", "n", "inference_20260101", "Recall@10,MRR@10,NDCG@10", ""]
    )

    assert result.command == (
        "uv run stage evaluation dataset=toy "
        "stage.inference_run_id=inference_20260101 "
        """metrics='["Recall@10","MRR@10","NDCG@10"]'"""
    )
    assert result.overrides == (
        "dataset=toy",
        "stage.inference_run_id=inference_20260101",
        """metrics=["Recall@10","MRR@10","NDCG@10"]""",
    )


def test_render_inference_command_accepts_prepared_input_mapping_name() -> None:
    command = render_command(
        "inference",
        [
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dummy_keyword"),
            HydraOverride("runtime=cpu"),
            HydraOverride("input_mapping=toy_dev"),
        ],
    )

    assert command.endswith("input_mapping=toy_dev")


def test_configure_flow_switches_nested_component_choice_with_mounted_override(
    tmp_path: Path,
) -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "2",
            "3",
            "y",
            "7",
            "2",
            "1",
            "0",
            "1",
            "",
        ],
        indexes_dir=tmp_path / "indexes",
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "runtime=gpu "
        "selections/embedding_model=e5/small_v2 "
        "component/query_preprocessor@pipeline.components.query_preprocessor=prefix_cleanup "
        "selections.index_id=index-1"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "runtime=gpu",
        "selections/embedding_model=e5/small_v2",
        "component/query_preprocessor@pipeline.components.query_preprocessor=prefix_cleanup",
        "selections.index_id=index-1",
    )


def test_configure_flow_edits_nested_selection_field(tmp_path: Path) -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "2",
            "3",
            "y",
            "6",
            "3",
            "8",
            "256",
            "0",
            "0",
            "1",
            "",
        ],
        indexes_dir=tmp_path / "indexes",
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "runtime=gpu "
        "selections/embedding_model=e5/small_v2 "
        "selections.embedding_model.tokenizer_kwargs.model_max_length=256 "
        "selections.index_id=index-1"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "runtime=gpu",
        "selections/embedding_model=e5/small_v2",
        "selections.embedding_model.tokenizer_kwargs.model_max_length=256",
        "selections.index_id=index-1",
    )


def test_configure_flow_shows_updated_value_after_field_edit(tmp_path: Path) -> None:
    result, output = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "2",
            "3",
            "y",
            "6",
            "3",
            "2",
            "my_e5",
            "2",
            "",
            "0",
            "0",
            "1",
            "",
        ],
        include_output=True,
        indexes_dir=tmp_path / "indexes",
    )

    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "runtime=gpu",
        "selections/embedding_model=e5/small_v2",
        "selections.embedding_model.artifact_name=my_e5",
        "selections.index_id=index-1",
    )
    assert any(
        "selections.embedding_model.artifact_name = my_e5" in line for line in output
    )
    assert "  1. index-1" in output


def test_configure_flow_can_list_pipeline_fields_with_list_values(
    tmp_path: Path,
) -> None:
    result, output = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "2",
            "3",
            "y",
            "4",
            "3",
            "0",
            "0",
            "1",
            "",
        ],
        include_output=True,
        indexes_dir=tmp_path / "indexes",
    )

    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "runtime=gpu",
        "selections/embedding_model=e5/small_v2",
        "selections.index_id=index-1",
    )
    assert any("pipeline.connections = " in line for line in output)
    assert "  0. Done" in output


def _run_with_answers(
    answers: list[str],
    *,
    include_output: bool = False,
    indexes_dir: Path | None = None,
):
    remaining = list(answers)
    output: list[str] = []

    if indexes_dir is not None:
        index_dir = indexes_dir / "index-1"
        index_dir.mkdir(parents=True)
        (index_dir / "index.jsonl").touch()

    def input_fn(prompt: str) -> str:
        output.append(prompt)
        return remaining.pop(0)

    result = run_configure(
        input_fn=input_fn,
        output_fn=output.append,
        config_dir=CONFIG_DIR,
        indexes_dir=indexes_dir,
    )

    assert not remaining
    assert "Command:" in output
    if include_output:
        return result, output
    return result
