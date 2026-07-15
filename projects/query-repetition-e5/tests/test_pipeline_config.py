from pathlib import Path

from retrieval_core.config import compose_stage_config
from retrieval_core.pipelines import load_async_pipeline


CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_stage_config(
        "inference",
        [
            "dataset=toy",
            "pipeline/inference@pipeline=dense_query_repetition",
            "selections/embedding_model=e5/small_v2",
        ],
        config_dir=CONFIG_DIR,
    )

    assert cfg.pipeline.components.query_repeater.type.endswith("QueryRepeater")
    load_async_pipeline(cfg.pipeline)
