# Dimensionality Reduction on Synthetic Data: PCA, t-SNE, and UMAP

This project generates synthetic 3D data with four clusters and reduces it to 2D with three methods:

- PCA
- t-SNE
- UMAP

The goal is to compare how each method represents the same 3D cluster structure in 2D.

---

## Setup

```python
!python -m pip install numpy==2.2.0 pandas==2.2.3 matplotlib==3.9.3 plotly==5.24.1
!python -m pip install --upgrade scikit-learn umap-learn
```

---

## Data

The data is created with `make_blobs`: 500 samples, 3 features, 4 clusters.

- Centers: `[2, -6, -6]`, `[-1, 9, 4]`, `[-8, 7, 2]`, `[4, 7, 9]`
- `cluster_std = [1, 1, 2, 3.5]`

So the first two clusters are compact, the third is more spread out, and the fourth is the most spread out.

![3D Scatter Plot of Four Blobs](3D.png)

The features are standardized with `StandardScaler` (`z = (x - mu) / sigma`) before reduction, so no feature dominates the variance or distance calculations.

---

## Methods

### PCA

Linear reduction. PCA finds the eigenvectors of the covariance matrix `Sigma = (1 / (n - 1)) X^T X`. The first component `w_1` maximizes `Var(X w_1)` with `||w_1|| = 1`, the second is the next best orthogonal direction. Each point is projected as `[x · w_1, x · w_2]`.

PCA keeps the large-scale variance structure but not non-linear neighborhoods.

### t-SNE

Non-linear reduction mainly used for visualization. It turns distances in the original space into probabilities:

`p_{j|i} = exp(-||x_i - x_j||^2 / (2 sigma_i^2)) / sum_{k != i} exp(-||x_i - x_k||^2 / (2 sigma_i^2))`

symmetrized as `p_{ij} = (p_{j|i} + p_{i|j}) / (2n)`. In 2D it uses a Student-t distribution:

`q_{ij} = (1 + ||y_i - y_j||^2)^(-1) / sum_{k != l} (1 + ||y_k - y_l||^2)^(-1)`

and minimizes `KL(P || Q) = sum p_{ij} log(p_{ij} / q_{ij})`. The heavy tails reduce crowding. Local neighborhoods are preserved well, but global distances should be read carefully.

Parameters: `n_components=2`, `perplexity=30`, `max_iter=1000`, `random_state=42`. Perplexity sets the effective neighborhood size.

### UMAP

Non-linear reduction based on a weighted neighborhood graph. It finds a 2D embedding whose graph is as close as possible to the original one, keeping nearby points close without collapsing unrelated points together.

Parameters: `n_components=2`, `random_state=42`, `min_dist=0.5`, `spread=1`, `n_jobs=1`. Smaller `min_dist` gives tighter clusters.

---

## Results

### t-SNE

![2D t-SNE Projection](2D-tSNE.png)

The four clusters stay clearly separated, and the distances between them are consistent with how far apart they were in 3D.

### UMAP

![2D UMAP Projection](2D-UMAP.png)

UMAP also separates the four clusters. The layout differs from t-SNE because it optimizes a different objective.

### PCA

![2D PCA Projection](2D-PCA.png)

PCA separates most clusters even though it is linear, and it keeps the relative densities of the blobs. Some compression is expected when projecting 3D to 2D.

PCA and t-SNE ran quickly compared to UMAP.

---

## Comparison

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Type | Linear | Non-linear | Non-linear |
| Preserves | Variance, relative density | Local neighborhoods | Graph neighborhoods, some global layout |
| Speed here | Fast | Fast | Slower |
| Interpretable axes | Yes | No | No |
| Sensitive to parameters | Little | Yes | Yes |

---

## Files

- `tSNE-and-UMAP.ipynb`: main notebook
- `3D.png`, `2D-tSNE.png`, `2D-UMAP.png`, `2D-PCA.png`: figures
