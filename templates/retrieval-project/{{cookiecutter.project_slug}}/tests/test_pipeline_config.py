from pathlib import Path

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline={{ cookiecutter.pipeline_name }}",
            "selections/embedding_model={{ cookiecutter.embedding_model }}",
        ],
        config_dir=CONFIG_DIR,
    )

    assert cfg.pipeline.components.query_transformer.type.endswith(
        "{{ cookiecutter.component_class_name }}"
    )
    load_async_pipeline(cfg.pipeline)
