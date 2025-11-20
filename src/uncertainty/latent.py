# src/uncertainty/latent.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class LatentUncertaintyResult:
    """
    Container for latent-space uncertainty metrics for a single prompt/model.
    """
    mean_cosine_similarity: float
    var_cosine_similarity: float
    mean_dim_variance: float


def _to_flat_tensor(latent: TensorLike) -> torch.Tensor:
    """
    Convert a latent (torch.Tensor or numpy array) into a 1D float32 torch tensor.
    Expected shape is (C, H, W) or (B, C, H, W); if batched, we take the first element.
    """
    if isinstance(latent, np.ndarray):
        t = torch.from_numpy(latent)
    else:
        t = latent

    if t.ndim == 4:
        # Assume (B, C, H, W) -> take first in batch
        t = t[0]
    elif t.ndim != 3:
        raise ValueError(f"Expected latent of shape (C,H,W) or (B,C,H,W), got {t.shape}")

    return t.reshape(-1).float().cpu()  # (D,)


def compute_latent_uncertainty(
    latents: List[TensorLike],
) -> LatentUncertaintyResult:
    """
    Compute latent-space uncertainty metrics across multiple seeds for the same prompt.

    Args:
        latents: list of latent tensors/arrays, one per seed.
                 Each item is typically shape (4, H, W) or (1, 4, H, W).

    Returns:
        LatentUncertaintyResult with:
            - mean_cosine_similarity: mean pairwise cosine similarity in latent space
            - var_cosine_similarity: variance of pairwise cosine similarity
            - mean_dim_variance: mean variance per latent dimension across seeds
    """
    n = len(latents)
    if n < 2:
        # Not enough samples to define pairwise metrics
        return LatentUncertaintyResult(
            mean_cosine_similarity=1.0,
            var_cosine_similarity=0.0,
            mean_dim_variance=0.0,
        )

    # Flatten all latents into (N, D)
    flat_list = [_to_flat_tensor(z) for z in latents]
    mat = torch.stack(flat_list, dim=0)  # (N, D)

    # Compute per-dimension variance across seeds
    #    mat: (N, D) -> var over N -> (D,)
    dim_var = mat.var(dim=0, unbiased=False)  # avoid tiny sample-size correction
    mean_dim_variance = float(dim_var.mean().item())

    # Compute pairwise cosine similarities in latent space
    #    Normalize each vector then S = F F^T
    eps = 1e-8
    norms = mat.norm(dim=1, keepdim=True).clamp_min(eps)
    feats = mat / norms  # (N, D), unit norm

    # Similarity matrix S[i,j] = cos(latent_i, latent_j)
    S = feats @ feats.T  # (N, N)

    # Extract upper triangular (i<j) to avoid self-similarity and duplicates
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    sims = S[idx_i, idx_j]  # (N*(N-1)/2,)

    mean_cos_sim = float(sims.mean().item())
    var_cos_sim = float(sims.var(unbiased=False).item())

    return LatentUncertaintyResult(
        mean_cosine_similarity=mean_cos_sim,
        var_cosine_similarity=var_cos_sim,
        mean_dim_variance=mean_dim_variance,
    )
