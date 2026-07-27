from pathlib import Path

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline


PROJECT_DIR = Path(__file__).parents[1]


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "runtime=cpu",
            "pipeline/inference@pipeline={{ cookiecutter.package_name }}/{{ cookiecutter.pipeline_name }}",
            "selections/embedding_model=e5/small_v2",
            "selections.index_id=test-index",
            "stage.run_id=test-inference",
        ],
        project_dir=PROJECT_DIR,
    )

    assert cfg.pipeline.components.query_transformer.type.endswith(
        "{{ cookiecutter.component_class_name }}"
    )
    assert cfg.stage.run_id == "test-inference"
    load_async_pipeline(cfg.pipeline)
