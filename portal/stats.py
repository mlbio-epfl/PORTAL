"""Gaussian statistics for the PORTAL transport map.

Both the vision (source) and language (target) sides are summarized by the same
triple: mean, top-k PCA directions, and the spread along each direction.
"""

import math

import torch
from torch import Tensor

__all__ = ["pca_stats", "Stats"]

# (mu, V_k, sigma_k): mean (d,), top-k PCA directions (k, d) with orthonormal
# rows, and the std along each principal axis (k,).
Stats = tuple[Tensor, Tensor, Tensor]


def pca_stats(features: Tensor, k: int) -> Stats:
    """Mean, top-k PCA basis, and per-axis spread of a feature cloud.

    `features` is (N, d), or (N, P, d) which is flattened to (N*P, d). Returns
    (mu, V_k, sigma_k); the rank shrinks below k when N or d is the limit.
    """
    x = features.float()
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    if x.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape {tuple(x.shape)}")

    n = x.shape[0]
    mu = x.mean(dim=0)
    # sigma_i = S_i / sqrt(N-1) converts singular values to covariance stds.
    _, s, vt = torch.linalg.svd(x - mu, full_matrices=False)
    k_eff = min(k, s.shape[0])
    sigma = s[:k_eff] / math.sqrt(max(n - 1, 1))
    return mu, vt[:k_eff], sigma
