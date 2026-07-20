from pathlib import Path

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "{{ cookiecutter.project_slug }}"


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_stage_config(
        "runs/treatment",
        experiment_dir=EXPERIMENT_DIR,
    )

    assert cfg.pipeline.components.query_transformer.type.endswith(
        "{{ cookiecutter.component_class_name }}"
    )
    assert cfg.stage.run_id.endswith("--treatment")
    load_async_pipeline(cfg.pipeline)
