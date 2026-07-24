from pathlib import Path

import pytest
import torch

from sparse_autoencoder_retrieval.model import (
    CompositeCodeSparseAutoencoder,
    load_autoencoder_checkpoint,
    save_autoencoder_checkpoint,
)
from sparse_autoencoder_retrieval.training import train_autoencoder


def test_eval_codes_are_deterministic_and_one_hot_per_codebook() -> None:
    model = CompositeCodeSparseAutoencoder(
        input_dim=3,
        num_codebooks=2,
        codebook_size=4,
        use_batch_norm=False,
    )
    model.eval()
    dense = torch.tensor([[1.0, 0.5, -0.25], [-1.0, 0.0, 1.0]])

    first = model.encode(dense)
    second = model.encode(dense)

    assert torch.equal(first, second)
    assert first.shape == (2, 8)
    assert torch.equal(first.reshape(2, 2, 4).sum(dim=-1), torch.ones(2, 2))
    assert set(first.unique().tolist()) <= {0.0, 1.0}


def test_loss_is_differentiable_and_reports_both_objectives() -> None:
    model = CompositeCodeSparseAutoencoder(
        input_dim=3,
        num_codebooks=2,
        codebook_size=3,
        use_batch_norm=False,
    )
    model.train()

    loss, metrics = model.loss(torch.randn(6, 3), balance_weight=0.5)
    loss.backward()

    assert set(metrics) == {"loss", "reconstruction_loss", "uniformity_loss"}
    assert model.encoder.weight.grad is not None
    assert model.decoder.weight.grad is not None


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = CompositeCodeSparseAutoencoder(
        input_dim=3,
        num_codebooks=2,
        codebook_size=4,
        use_batch_norm=False,
    )
    path = save_autoencoder_checkpoint(model, tmp_path / "ccsa.pt")

    restored = load_autoencoder_checkpoint(path)

    assert restored.checkpoint_config() == model.checkpoint_config()
    assert all(
        torch.equal(restored.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )
    assert not restored.training


def test_training_runs_on_frozen_dense_embeddings() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    model, history = train_autoencoder(
        embeddings,
        num_codebooks=2,
        codebook_size=2,
        epochs=2,
        batch_size=4,
        use_batch_norm=False,
        seed=7,
    )

    assert model.input_dim == 3
    assert [entry["epoch"] for entry in history] == [1.0, 2.0]
    assert all(entry["loss"] >= 0 for entry in history)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0, "num_codebooks": 2, "codebook_size": 2}, "input_dim"),
        ({"input_dim": 3, "num_codebooks": 0, "codebook_size": 2}, "num_codebooks"),
        ({"input_dim": 3, "num_codebooks": 2, "codebook_size": 1}, "codebook_size"),
    ],
)
def test_model_rejects_invalid_architecture(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CompositeCodeSparseAutoencoder(**kwargs)
