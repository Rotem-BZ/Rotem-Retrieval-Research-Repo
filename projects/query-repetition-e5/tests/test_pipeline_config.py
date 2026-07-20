from pathlib import Path

from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.pipelines import load_async_pipeline


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments" / "query-repetition-e5-small-scifact"


def test_project_pipeline_composes_with_core_config_groups() -> None:
    cfg = compose_stage_config(
        "runs/repeated",
        experiment_dir=EXPERIMENT_DIR,
    )

    assert cfg.pipeline.components.query_repeater.type.endswith("QueryRepeater")
    assert cfg.stage.run_id.endswith("--repeated")
    load_async_pipeline(cfg.pipeline)
