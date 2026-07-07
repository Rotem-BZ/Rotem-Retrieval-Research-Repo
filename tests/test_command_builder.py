from pathlib import Path

from retrieval_research.command_builder import (
    HydraOverride,
    collect_selected_configs,
    discover_config_choices,
    editable_fields,
    effective_editable_fields,
    extract_required_defaults,
    render_command,
    run_configure,
    validate_config,
)


CONFIG_DIR = Path("configs")


def test_discovers_recursive_config_choices() -> None:
    choices = discover_config_choices("selections/embedding_model", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == [
        "e5/base_v2",
        "e5/large_v2",
        "e5/small_v2",
    ]


def test_discovers_reranker_model_choices() -> None:
    choices = discover_config_choices("selections/reranker_model", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == ["bge/v2_m3"]


def test_dataset_choices_do_not_include_nested_mapping_configs() -> None:
    choices = discover_config_choices("dataset", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == [
        "beir_scifact",
        "beir_scifact_smoke",
        "toy",
    ]


def test_discovers_builtin_input_mapping_choices() -> None:
    choices = discover_config_choices("input_mapping", config_dir=CONFIG_DIR)

    assert [choice.name for choice in choices] == [
        "full",
        "dev_tiny",
        "judged_only",
        "random_smoke",
    ]


def test_extracts_required_defaults_from_stage_and_pipeline_configs() -> None:
    stage_required = extract_required_defaults(CONFIG_DIR / "indexing.yaml")
    pipeline_required = extract_required_defaults(
        CONFIG_DIR / "pipeline" / "indexing" / "dense_jsonl.yaml"
    )

    assert [(item.group, item.override_key) for item in stage_required] == [
        ("dataset", "dataset"),
        ("pipeline/indexing", "pipeline/indexing@pipeline"),
    ]
    assert [(item.group, item.override_key) for item in pipeline_required] == [
        ("selections/embedding_model", "selections/embedding_model"),
    ]


def test_collects_selected_config_graph_with_mount_prefixes() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
        ],
        config_dir=CONFIG_DIR,
    )

    assert [(item.label, item.field_prefix, item.status) for item in selected] == [
        ("stage inference", "", "stage"),
        ("paths=local", "paths", "default"),
        ("dataset=toy", "dataset", "selected"),
        ("input_mapping=full", "input_mapping", "default"),
        ("pipeline/inference@pipeline=dense_jsonl", "pipeline", "selected"),
        ("selections/embedding_model=e5/small_v2", "selections.embedding_model", "selected"),
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
    ]


def test_editable_fields_render_against_mounted_package_path() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
        ],
        config_dir=CONFIG_DIR,
    )
    embedding_model = selected[5]

    fields = editable_fields(embedding_model)

    assert fields[-1].path == "selections.embedding_model.tokenizer_kwargs.model_max_length"
    assert fields[-1].value == 512


def test_effective_editable_fields_reflect_current_overrides() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
            HydraOverride("selections.embedding_model.artifact_name=my_e5"),
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
        ],
    )

    artifact_name = next(
        field for field in fields if field.path == "selections.embedding_model.artifact_name"
    )
    assert artifact_name.value == "my_e5"


def test_effective_editable_fields_convert_composed_list_values() -> None:
    selected = collect_selected_configs(
        stage_name="inference",
        stage_path=CONFIG_DIR / "inference.yaml",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
        ],
        config_dir=CONFIG_DIR,
    )
    pipeline = selected[4]

    fields = effective_editable_fields(
        pipeline,
        stage_name="inference",
        overrides=[
            HydraOverride("dataset=toy"),
            HydraOverride("pipeline/inference@pipeline=dense_jsonl"),
            HydraOverride("selections/embedding_model=e5/small_v2"),
        ],
    )

    connections = next(field for field in fields if field.path == "pipeline.connections")
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
        ],
        dry_run=True,
    )

    assert (
        command
        == "uv run stage --dry-run inference dataset=toy "
        "pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2"
    )


def test_configure_flow_builds_indexing_dummy_command() -> None:
    result = _run_with_answers(["2", "3", "3", "n", "n", "toy_index", ""])

    assert result.command == (
        "uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl "
        "stage.run_name=toy_index"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/indexing@pipeline=dummy_jsonl",
        "stage.run_name=toy_index",
    )


def test_configure_flow_builds_inference_dense_command_with_top_k() -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "3",
            "n",
            "n",
            "toy_dense",
            "pipeline.components.retriever.init_parameters.top_k=100",
            "",
        ]
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "selections/embedding_model=e5/small_v2 "
        "stage.run_name=toy_dense "
        "pipeline.components.retriever.init_parameters.top_k=100"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "stage.run_name=toy_dense",
        "pipeline.components.retriever.init_parameters.top_k=100",
    )


def test_configure_flow_builds_evaluation_command_with_metrics() -> None:
    result = _run_with_answers(["1", "3", "n", "n", "Recall@10,MRR@10,NDCG@10", ""])

    assert result.command == (
        "uv run stage evaluation dataset=toy "
        """metrics='["Recall@10","MRR@10","NDCG@10"]'"""
    )
    assert result.overrides == (
        "dataset=toy",
        """metrics=["Recall@10","MRR@10","NDCG@10"]""",
    )


def test_dense_e5_wizard_output_composes() -> None:
    result = _run_with_answers(["3", "3", "4", "3", "n", "n", "toy_dense", ""])

    validate_config(result.stage_name, result.overrides)


def test_configure_flow_switches_default_input_mapping_choice() -> None:
    result = _run_with_answers(["3", "3", "5", "y", "4", "2", "2", "6", "n", "toy_keyword", ""])

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dummy_keyword "
        "input_mapping=dev_tiny stage.run_name=toy_keyword"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dummy_keyword",
        "input_mapping=dev_tiny",
        "stage.run_name=toy_keyword",
    )


def test_configure_flow_switches_nested_component_choice_with_mounted_override() -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "3",
            "y",
            "7",
            "2",
            "1",
            "10",
            "n",
            "toy_dense",
            "",
        ]
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "selections/embedding_model=e5/small_v2 "
        "component/query_preprocessor@pipeline.components.query_preprocessor=prefix_cleanup "
        "stage.run_name=toy_dense"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "component/query_preprocessor@pipeline.components.query_preprocessor=prefix_cleanup",
        "stage.run_name=toy_dense",
    )


def test_configure_flow_edits_nested_selection_field() -> None:
    result = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "3",
            "y",
            "6",
            "3",
            "8",
            "256",
            "9",
            "10",
            "n",
            "toy_dense",
            "",
        ]
    )

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "selections/embedding_model=e5/small_v2 "
        "selections.embedding_model.tokenizer_kwargs.model_max_length=256 "
        "stage.run_name=toy_dense"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "selections.embedding_model.tokenizer_kwargs.model_max_length=256",
        "stage.run_name=toy_dense",
    )


def test_configure_flow_shows_updated_value_after_field_edit() -> None:
    result, output = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "3",
            "y",
            "6",
            "3",
            "2",
            "my_e5",
            "2",
            "",
            "9",
            "10",
            "n",
            "toy_dense",
            "",
        ],
        include_output=True,
    )

    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "selections.embedding_model.artifact_name=my_e5",
        "stage.run_name=toy_dense",
    )
    assert any("selections.embedding_model.artifact_name = my_e5" in line for line in output)


def test_configure_flow_can_list_pipeline_fields_with_list_values() -> None:
    result, output = _run_with_answers(
        [
            "3",
            "3",
            "4",
            "3",
            "y",
            "5",
            "3",
            "5",
            "10",
            "n",
            "toy_dense",
            "",
        ],
        include_output=True,
    )

    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "stage.run_name=toy_dense",
    )
    assert any("pipeline.connections = " in line for line in output)


def _run_with_answers(answers: list[str], *, include_output: bool = False):
    remaining = list(answers)
    output: list[str] = []

    def input_fn(prompt: str) -> str:
        output.append(prompt)
        return remaining.pop(0)

    result = run_configure(input_fn=input_fn, output_fn=output.append, config_dir=CONFIG_DIR)

    assert not remaining
    assert "Command:" in output
    if include_output:
        return result, output
    return result
