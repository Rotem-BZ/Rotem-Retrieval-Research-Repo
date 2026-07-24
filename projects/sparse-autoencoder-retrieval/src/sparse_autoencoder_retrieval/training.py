"""Train CCSA over frozen dense embeddings stored in a repository JSONL index."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from sparse_autoencoder_retrieval.model import (
    CompositeCodeSparseAutoencoder,
    save_autoencoder_checkpoint,
)


def load_dense_embeddings(input_path: str | Path) -> torch.Tensor:
    """Load a rectangular dense embedding matrix from a JSONL document index."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dense embedding index not found: {path}")
    embeddings: list[list[float]] = []
    expected_dim: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            embedding = record.get("embedding")
            if embedding is None:
                raise ValueError(f"Record on line {line_number} has no dense embedding.")
            values = [float(value) for value in embedding]
            expected_dim = expected_dim or len(values)
            if len(values) != expected_dim:
                raise ValueError(
                    f"Embedding dimension changed on line {line_number}: "
                    f"expected {expected_dim}, received {len(values)}."
                )
            embeddings.append(values)
    if len(embeddings) < 2:
        raise ValueError("CCSA training requires at least two dense embeddings.")
    return torch.tensor(embeddings, dtype=torch.float32)


def train_autoencoder(
    embeddings: torch.Tensor,
    *,
    num_codebooks: int,
    codebook_size: int,
    epochs: int = 20,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    balance_weight: float = 1.0,
    temperature: float = 1.0,
    use_batch_norm: bool = True,
    device: str = "cpu",
    seed: int = 13,
) -> tuple[CompositeCodeSparseAutoencoder, list[dict[str, float]]]:
    """Fit CCSA and return the trained model plus per-epoch metrics."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] == 0:
        raise ValueError("embeddings must have shape [at least 2 examples, positive dimension].")
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two when batch normalization is used.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if balance_weight < 0:
        raise ValueError("balance_weight must be non-negative.")

    random.seed(seed)
    torch.manual_seed(seed)
    model = CompositeCodeSparseAutoencoder(
        input_dim=embeddings.shape[1],
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        temperature=temperature,
        use_batch_norm=use_batch_norm,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    effective_batch_size = min(batch_size, embeddings.shape[0])
    drop_last = use_batch_norm and embeddings.shape[0] % effective_batch_size == 1
    loader = DataLoader(
        TensorDataset(embeddings),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
        generator=generator,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "uniformity_loss": 0.0,
        }
        example_count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.loss(batch, balance_weight=balance_weight)
            loss.backward()
            optimizer.step()
            for name in totals:
                totals[name] += float(metrics[name]) * batch.shape[0]
            example_count += batch.shape[0]
        history.append(
            {
                "epoch": float(epoch),
                **{name: value / example_count for name, value in totals.items()},
            }
        )
    model.eval()
    return model, history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a composite-code sparse autoencoder over a dense JSONL index."
    )
    parser.add_argument("--input-path", required=True, help="Dense JSONL index with embeddings.")
    parser.add_argument("--output-path", required=True, help="Destination .pt checkpoint.")
    parser.add_argument("--num-codebooks", type=int, required=True)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for offline CCSA fitting."""

    args = _parser().parse_args(argv)
    embeddings = load_dense_embeddings(args.input_path)
    model, history = train_autoencoder(
        embeddings,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        balance_weight=args.balance_weight,
        temperature=args.temperature,
        use_batch_norm=not args.no_batch_norm,
        device=args.device,
        seed=args.seed,
    )
    metadata: dict[str, Any] = {
        "training_input": str(Path(args.input_path).resolve()),
        "training_examples": int(embeddings.shape[0]),
        "history": history,
        "balance_weight": args.balance_weight,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    path = save_autoencoder_checkpoint(
        model,
        args.output_path,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    print(json.dumps({"checkpoint_path": str(path), "final_metrics": history[-1]}))
