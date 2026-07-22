from pathlib import Path

import pytest

from retrieval_core.utils.config import compose_entrypoint_config
from retrieval_core.utils.pipelines import load_async_pipeline


PROJECT_DIR = Path(__file__).parents[1]
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
RUN_CONFIGS = sorted(EXPERIMENTS_DIR.glob("*/configs/runs/*.yaml"))


@pytest.mark.parametrize(
    "entrypoint",
    RUN_CONFIGS,
    ids=lambda path: f"{path.parents[2].name}/{path.stem}",
)
def test_experiment_run_config_composes(entrypoint: Path) -> None:
    cfg = compose_entrypoint_config(entrypoint)

    expected_run_id = f"{entrypoint.parents[2].name}--{entrypoint.stem}"
    assert cfg.stage.run_id == expected_run_id
    if "pipeline" in cfg:
        load_async_pipeline(cfg.pipeline)


@pytest.mark.parametrize(
    "entrypoint",
    sorted(
        (EXPERIMENTS_DIR / "chonkie-chunkers-scifact" / "configs" / "runs").glob(
            "*-indexing.yaml"
        )
    ),
    ids=lambda path: path.stem,
)
def test_chonkie_indexing_configs_keep_source_document_identity(entrypoint: Path) -> None:
    cfg = compose_entrypoint_config(entrypoint)

    if entrypoint.stem == "baseline-indexing":
        assert "source_id_adapter" not in cfg.pipeline.components
    else:
        assert cfg.pipeline.components.source_id_adapter.type.endswith("SourceDocumentIdAdapter")


def test_all_expected_integration_classes_are_represented() -> None:
    configured_types = {
        path.read_text(encoding="utf-8").splitlines()[0].removeprefix("type: ")
        for path in (PROJECT_DIR / "configs" / "component").rglob("*.yaml")
    }

    expected_suffixes = {
        "FastembedDocumentEmbedder",
        "FastembedTextEmbedder",
        "FastembedSparseDocumentEmbedder",
        "FastembedSparseTextEmbedder",
        "FastembedRanker",
        "FastembedLateInteractionRanker",
        "ChonkieTokenDocumentSplitter",
        "ChonkieSentenceDocumentSplitter",
        "ChonkieRecursiveDocumentSplitter",
        "ChonkieSemanticDocumentSplitter",
    }
    assert expected_suffixes <= {type_name.rsplit(".", 1)[-1] for type_name in configured_types}
