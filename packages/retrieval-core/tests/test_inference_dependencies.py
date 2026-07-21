from pathlib import Path

import pytest
from omegaconf import OmegaConf

from retrieval_core.stages.inference import prepare_inference_config
from retrieval_core.utils.artifacts import discover_index_ids


def test_candidate_only_inference_has_no_index_dependency() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "metadata": {"description": "reranker"},
                "components": {"ranker": {"init_parameters": {}}},
            }
        }
    )

    prepare_inference_config(cfg)

    assert "index_path" not in cfg.pipeline.components.ranker.init_parameters


def test_index_backed_inference_requires_an_index_id(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"indexes_dir": str(tmp_path / "indexes")},
            "selections": {"index_id": None},
            "pipeline": {
                "components": {
                    "retriever": {
                        "init_parameters": {
                            "index_path": "${paths.indexes_dir}/${selections.index_id}/index.jsonl"
                        }
                    }
                },
            },
        }
    )

    with pytest.raises(ValueError, match="requires a non-empty selections.index_id"):
        prepare_inference_config(cfg)


def test_index_backed_inference_uses_canonical_selected_index(tmp_path: Path) -> None:
    index_path = tmp_path / "indexes" / "index-1" / "index.jsonl"
    index_path.parent.mkdir(parents=True)
    index_path.touch()
    cfg = OmegaConf.create(
        {
            "paths": {"indexes_dir": str(tmp_path / "indexes")},
            "selections": {"index_id": "index-1"},
            "pipeline": {
                "components": {
                    "retriever": {
                        "init_parameters": {
                            "index_path": "${paths.indexes_dir}/${selections.index_id}/index.jsonl"
                        }
                    }
                },
            },
        }
    )

    prepare_inference_config(cfg)

    assert Path(cfg.pipeline.components.retriever.init_parameters.index_path) == index_path


def test_index_backed_inference_rejects_missing_selected_index(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"indexes_dir": str(tmp_path / "indexes")},
            "selections": {"index_id": "missing"},
            "pipeline": {
                "components": {
                    "retriever": {
                        "init_parameters": {
                            "index_path": "${paths.indexes_dir}/${selections.index_id}/index.jsonl"
                        }
                    }
                },
            },
        }
    )

    with pytest.raises(FileNotFoundError, match="No index exists with selections.index_id"):
        prepare_inference_config(cfg)


def test_discovers_only_completed_canonical_indexes(tmp_path: Path) -> None:
    completed = tmp_path / "indexes" / "completed"
    completed.mkdir(parents=True)
    (completed / "index.jsonl").touch()
    (tmp_path / "indexes" / "incomplete").mkdir()
    loose_file = tmp_path / "indexes" / "not-a-directory"
    loose_file.touch()

    assert discover_index_ids(tmp_path / "indexes") == ["completed"]
