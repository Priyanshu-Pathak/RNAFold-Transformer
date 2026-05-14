# Methodology

## Input construction

For every central residue, the pipeline extracts a local sequence window of 21 nucleotides:

```text
window = 10 residues left + 1 center residue + 10 residues right
```

Each nucleotide position is represented using 8 features:

1. 4-dimensional nucleotide one-hot vector for A, C, G, U.
2. 4-dimensional MSA nucleotide-frequency profile for A, C, G, U.

This gives one training example with shape:

```text
21 × 8
```

## Target construction

The true 3D coordinates are standardized globally using the training-label coordinates. For each 21-residue window, the model computes the pairwise Euclidean distance matrix:

```text
D[i,j] = ||coord_i - coord_j||_2
```

The target shape is:

```text
21 × 21
```

## Model

The model is a local distance predictor named `DistTransformer`.

```text
21×8 features
→ linear projection to d_model=64
→ learned positional embedding
→ 2-layer Transformer encoder
→ pairwise concatenation of residue embeddings
→ MLP distance head
→ symmetric 21×21 distance matrix
```

## Training objective

The model minimizes MSE between predicted and true pairwise distance matrices.

```text
Loss = mean((D_pred - D_true)^2)
```

Metrics reported:

- Distance RMSE
- Distance MAE
- Per-window RMSE / MAE
- Pearson correlation between flattened true and predicted distances

## Secondary-structure baseline

The repository also includes a simple Nussinov dynamic-programming implementation for local dot-bracket secondary-structure prediction and arc-diagram visualization. This is not the main 3D model; it is included as an interpretable baseline/visual aid.
