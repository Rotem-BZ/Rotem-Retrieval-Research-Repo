"""Composite-code sparse autoencoder used to compress dense retrieval vectors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CompositeCodeSparseAutoencoder(nn.Module):
    """Map dense vectors to one-hot codes in each of several codebooks.

    This is the CCSA architecture from Lassance, Formal, and Clinchant: batch
    normalization, a linear encoder, hard groupwise activations, and a linear
    decoder. Training uses straight-through Gumbel softmax; evaluation uses a
    deterministic argmax so repeated indexing runs produce identical codes.
    """

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        temperature: float = 1.0,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be greater than zero.")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than one.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.temperature = temperature
        self.use_batch_norm = use_batch_norm
        self.code_dim = num_codebooks * codebook_size

        self.input_normalizer: nn.Module
        if use_batch_norm:
            self.input_normalizer = nn.BatchNorm1d(input_dim)
        else:
            self.input_normalizer = nn.Identity()
        self.encoder = nn.Linear(input_dim, self.code_dim)
        self.decoder = nn.Linear(self.code_dim, input_dim)

    def encode(self, dense_vectors: Tensor, *, stochastic: bool | None = None) -> Tensor:
        """Encode a ``[batch, input_dim]`` tensor into C-hot binary codes."""

        self._validate_dense_vectors(dense_vectors)
        logits = self.encoder(self.input_normalizer(dense_vectors))
        grouped_logits = logits.reshape(-1, self.num_codebooks, self.codebook_size)
        use_stochastic_codes = self.training if stochastic is None else stochastic
        if use_stochastic_codes:
            return F.gumbel_softmax(
                grouped_logits,
                tau=self.temperature,
                hard=True,
                dim=-1,
            ).reshape(-1, self.code_dim)

        selected = grouped_logits.argmax(dim=-1, keepdim=True)
        codes = torch.zeros_like(grouped_logits)
        codes.scatter_(-1, selected, 1.0)
        return codes.reshape(-1, self.code_dim)

    def decode(self, sparse_codes: Tensor) -> Tensor:
        """Decode a ``[batch, code_dim]`` tensor back into dense space."""

        if sparse_codes.ndim != 2 or sparse_codes.shape[1] != self.code_dim:
            raise ValueError(
                f"Expected sparse codes with shape [batch, {self.code_dim}], "
                f"received {tuple(sparse_codes.shape)}."
            )
        return self.decoder(sparse_codes)

    def forward(self, dense_vectors: Tensor) -> tuple[Tensor, Tensor]:
        """Return sparse codes and their dense reconstructions."""

        sparse_codes = self.encode(dense_vectors)
        return sparse_codes, self.decode(sparse_codes)

    def loss(
        self,
        dense_vectors: Tensor,
        *,
        balance_weight: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return CCSA reconstruction plus posting-list balance loss."""

        if balance_weight < 0:
            raise ValueError("balance_weight must be non-negative.")
        sparse_codes, reconstructed = self(dense_vectors)
        reconstruction_loss = F.mse_loss(reconstructed, dense_vectors)

        batch_size = dense_vectors.shape[0]
        target_count = batch_size / self.codebook_size
        activation_counts = sparse_codes.sum(dim=0)
        uniformity_loss = torch.linalg.vector_norm(
            activation_counts - target_count
        ) / math.sqrt(batch_size)
        total_loss = reconstruction_loss + balance_weight * uniformity_loss
        return total_loss, {
            "loss": total_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "uniformity_loss": uniformity_loss.detach(),
        }

    def checkpoint_config(self) -> dict[str, int | float | bool]:
        """Return the serializable architecture required to reload this model."""

        return {
            "input_dim": self.input_dim,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size,
            "temperature": self.temperature,
            "use_batch_norm": self.use_batch_norm,
        }

    def _validate_dense_vectors(self, dense_vectors: Tensor) -> None:
        if dense_vectors.ndim != 2 or dense_vectors.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected dense vectors with shape [batch, {self.input_dim}], "
                f"received {tuple(dense_vectors.shape)}."
            )


def save_autoencoder_checkpoint(
    model: CompositeCodeSparseAutoencoder,
    output_path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist a portable CCSA checkpoint containing config and weights."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists and overwrite=false: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ccsa-v1",
            "config": model.checkpoint_config(),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def load_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> CompositeCodeSparseAutoencoder:
    """Load and validate a checkpoint written by :func:`save_autoencoder_checkpoint`."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Sparse autoencoder checkpoint not found at {path}. "
            "Train it with the train-ccsa command before indexing or retrieval."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "ccsa-v1":
        raise ValueError(f"Unsupported sparse autoencoder checkpoint format: {path}")
    config = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise TypeError(f"Malformed sparse autoencoder checkpoint: {path}")

    model = CompositeCodeSparseAutoencoder(**config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
