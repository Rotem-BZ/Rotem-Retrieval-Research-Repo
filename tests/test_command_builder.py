from pathlib import Path

from retrieval_research.command_builder import (
    HydraOverride,
    discover_config_choices,
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
    result = _run_with_answers(["2", "3", "3", "n", ""])

    assert result.command == (
        "uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl"
    )
    assert result.overrides == ("dataset=toy", "pipeline/indexing@pipeline=dummy_jsonl")


def test_configure_flow_builds_inference_dense_command_with_top_k() -> None:
    result = _run_with_answers(["3", "3", "2", "3", "n", "100", ""])

    assert result.command == (
        "uv run stage inference dataset=toy pipeline/inference@pipeline=dense_jsonl "
        "selections/embedding_model=e5/small_v2 retrieval.top_k=100"
    )
    assert result.overrides == (
        "dataset=toy",
        "pipeline/inference@pipeline=dense_jsonl",
        "selections/embedding_model=e5/small_v2",
        "retrieval.top_k=100",
    )


def test_configure_flow_builds_evaluation_command_with_metrics() -> None:
    result = _run_with_answers(["1", "3", "n", "Recall@10,MRR@10,NDCG@10", ""])

    assert result.command == (
        "uv run stage evaluation dataset=toy "
        """metrics='["Recall@10","MRR@10","NDCG@10"]'"""
    )
    assert result.overrides == (
        "dataset=toy",
        """metrics=["Recall@10","MRR@10","NDCG@10"]""",
    )


def test_dense_e5_wizard_output_composes() -> None:
    result = _run_with_answers(["3", "3", "2", "3", "n", "", ""])

    validate_config(result.stage_name, result.overrides)


def _run_with_answers(answers: list[str]):
    remaining = list(answers)
    output: list[str] = []

    def input_fn(prompt: str) -> str:
        output.append(prompt)
        return remaining.pop(0)

    result = run_configure(input_fn=input_fn, output_fn=output.append, config_dir=CONFIG_DIR)

    assert not remaining
    assert "Command:" in output
    return result
