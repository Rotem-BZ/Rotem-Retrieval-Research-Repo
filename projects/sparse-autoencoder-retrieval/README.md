# Sparse autoencoders for retrieval

This project tests whether a frozen dense bi-encoder can be converted into an
efficient learned sparse retriever. It implements the architecture from
[Composite Code Sparse Autoencoders for First-Stage Retrieval](https://arxiv.org/abs/2204.07023)
(CCSA) and wires it into this repository's Haystack v2 indexing and inference
stages.

CCSA is the closest match to the research question: it was designed to project
pre-existing Siamese/bi-encoder embeddings into balanced C-hot semantic codes and
retrieve them with posting lists. The model uses a one-layer encoder, hard
straight-through Gumbel softmax during training, deterministic groupwise argmax at
inference, a linear decoder, and the paper's posting-list uniformity regularizer.

## Existing implementations considered

- [Sentence Transformers sparse encoders](https://www.sbert.net/docs/quickstart.html#sparse-encoder)
  are production-quality SPLADE-style vocabulary encoders. They are a useful
  baseline, but they do not add an SAE to an existing dense bi-encoder.
- [OpenAI's sparse_autoencoder](https://github.com/openai/sparse_autoencoder)
  is a strong general SAE implementation for transformer activations. Its
  unstructured ReLU/Top-K latents do not implement CCSA's balanced composite
  codes or retrieval index.
- [CompresSAE](https://github.com/recombee/CompresSAE) is an MIT-licensed
  sparse embedding compressor optimized for recommender retrieval and is a good
  future weighted-Top-K treatment.
- [sporse](https://github.com/arclabs561/sporse) now includes CCSA and a sparse
  index in Rust. It is the closest existing end-to-end implementation, but it
  cannot be used directly as a Haystack Python component.

The code here is therefore a small native Python/Haystack adaptation of the
published CCSA equations, not a fork of those repositories.

## Architecture

```text
text -> Sentence Transformer bi-encoder -> dense vector (d)
     -> CCSA encoder -> C groups x L choices -> C-hot semantic sparse vector
     -> postings[latent dimension] -> [(document ordinal, weight), ...]
```

With the checked-in starting configuration, `C=32` and `L=256`: every vector has
exactly 32 active dimensions out of 8,192. The theoretical code is 256 bits per
document; the JSON research index deliberately uses readable postings instead of
bit packing. A well-balanced index has roughly `N / 256` documents in each
posting list.

The project owns four Haystack components:

| Component | Input | Output | Behavior |
| --- | --- | --- | --- |
| `SparseAutoencoderDocumentEmbedder` | `documents` | `documents` | Copies documents and attaches native Haystack `SparseEmbedding` values. |
| `SparseAutoencoderTextEmbedder` | `text` | `sparse_embedding` | Uses query-mode bi-encoder encoding followed by the same CCSA checkpoint. |
| `SemanticSparseIndexer` | `documents`, optional `append` | `index_path`, `indexed_count` | Writes document records and dimension-keyed posting lists. |
| `SemanticSparseRetriever` | `sparse_embedding`, optional candidates/top-k | `documents` | Accumulates sparse dot products over only the query's posting lists. |

Model and checkpoint loading are lazy and idempotent. The embedders do not mutate
input documents. Missing checkpoints, dense-dimension mismatches, missing sparse
vectors, duplicate document IDs, and malformed indexes fail with explicit errors.

## Setup

From this project directory:

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

The first real run downloads the configured Sentence Transformer checkpoint.
Tests replace that boundary with a local fake encoder and do not use the network.

## End-to-end experiment workflow

First prepare a dataset and build the normal dense E5-small index. This produces
the frozen corpus embeddings used to train the SAE:

```powershell
uv run prepare-beir --data-dir ../../data --dataset scifact
uv run stage indexing dataset=beir_scifact runtime=cpu pipeline/indexing@pipeline=dense/documents_in_memory selections/embedding_model=e5/small_v2 selections.index_id=scifact-e5-small-dense stage.run_id=scifact-e5-small-dense-indexing
```

Train CCSA over those embeddings. The output name agrees with the checked-in
`e5_small_ccsa` selection:

```powershell
uv run train-ccsa --input-path artifacts/indexes/scifact-e5-small-dense/index.json --output-path artifacts/models/e5-small-ccsa-c32-l256.pt --num-codebooks 32 --codebook-size 256 --epochs 20 --batch-size 1024 --balance-weight 1.0
```

The paper notes that larger batches estimate whole-index balance better. Start
with the largest batch that fits in memory, then treat `C`, `L`, balance weight,
and batch size as experimental factors.

Build the semantic sparse postings index with the same bi-encoder and checkpoint:

```powershell
uv run stage indexing dataset=beir_scifact runtime=cpu pipeline/indexing@pipeline=sparse_autoencoder_retrieval/semantic_sparse selections/embedding_model=e5/small_v2 selections/sparse_autoencoder=sparse_autoencoder_retrieval/e5_small_ccsa selections.index_id=scifact-e5-small-ccsa stage.run_id=scifact-e5-small-ccsa-indexing
```

Run retrieval:

```powershell
uv run stage inference dataset=beir_scifact runtime=cpu pipeline/inference@pipeline=sparse_autoencoder_retrieval/semantic_sparse selections/embedding_model=e5/small_v2 selections/sparse_autoencoder=sparse_autoencoder_retrieval/e5_small_ccsa selections.index_id=scifact-e5-small-ccsa stage.run_id=scifact-e5-small-ccsa-inference
```

For a fair baseline, run the core dense inference pipeline against
`scifact-e5-small-dense` with the same E5 selection and retrieval depth. Compare
at least Recall@1000, MRR@10, index bytes, mean active dimensions, posting-list
imbalance, query encoding time, and postings scoring time. CCSA is an approximate
first-stage method, so recall and latency are more diagnostic than only top-rank
precision.

## Index artifact

`SemanticSparseIndexer` writes a versioned JSON object containing:

- `documents`: content, metadata, score, and optional dense embedding;
- `postings`: sparse dimension to `[document_ordinal, weight]` pairs; and
- `statistics`: document, declared/active-dimension, and posting counts.

The indexer accepts later batches with `append: true`. It preserves global document
ordinals and duplicate-ID validation across those calls while the stage owns
temporary-artifact cleanup and atomic publication.

The default omits dense embeddings. Sparse vectors are reconstructed from the
posting lists when results are returned. The format favors inspectability and
deterministic experiments; a production follow-up should use compressed integer
posting lists, WAND/block-max scoring, or the Rust implementation noted above.
