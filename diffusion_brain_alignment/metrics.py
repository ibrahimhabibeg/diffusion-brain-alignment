"""
Core mathematical functions for Representational Similarity Analysis (RSA).
"""

import math

import numpy as np
from scipy.stats import rankdata
import torch


def calc_rdm_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the 1 - Pearson correlation Representational Dissimilarity Matrix (RDM).

    Equivalent to the correlation method in the rsatoolbox, but implemented in PyTorch for GPU acceleration.

    Args:
        x (torch.Tensor): A 2D tensor of shape (N_conditions, N_features).

    Returns:
        torch.Tensor: A symmetric (N_conditions, N_conditions) dissimilarity matrix.
    """
    x_centered = x - x.mean(dim=1, keepdim=True)
    x_norm = x_centered / (torch.norm(x_centered, p=2, dim=1, keepdim=True) + 1e-8)
    sim = torch.mm(x_norm, x_norm.t())
    return 1.0 - sim


def compute_rsa_score(rdm1: torch.Tensor, rdm2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman rank correlation (RSA score) between two RDMs.

    Equivalent to the Spearman's Rho method for comparing RDMs in the rsatoolbox, but implemented in PyTorch for GPU acceleration.

    Args:
        rdm1 (torch.Tensor): The first RDM matrix (e.g., Artificial).
        rdm2 (torch.Tensor): The second RDM matrix (e.g., Biological).

    Returns:
        torch.Tensor: A scalar tensor containing the Spearman correlation coefficient.
    """
    assert rdm1.device == rdm2.device, "Both RDMs must be on the same device."
    assert rdm1.shape == rdm2.shape, "Both RDMs must have the same shape."

    num_conditions = rdm2.shape[0]

    i_upper, j_upper = torch.triu_indices(
        num_conditions, num_conditions, offset=1, device=rdm1.device
    )

    vec1 = rdm1[i_upper, j_upper]
    vec2 = rdm2[i_upper, j_upper]

    ranks1 = torch.argsort(torch.argsort(vec1)).float()
    ranks2 = torch.argsort(torch.argsort(vec2)).float()

    ranks1_centered = ranks1 - torch.mean(ranks1)
    ranks2_centered = ranks2 - torch.mean(ranks2)

    n = vec1.shape[0]
    expected_variance = ((n**3) - n) / 12.0
    expected_std = math.sqrt(expected_variance)

    ranks1_scaled = ranks1_centered / expected_std
    ranks2_scaled = ranks2_centered / expected_std

    correlation = torch.dot(ranks1_scaled, ranks2_scaled)
    return correlation


def transform_to_percentile_rdm(rdm: np.ndarray) -> np.ndarray:
    """
    Transforms raw RDM dissimilarity values into percentile ranks [0, 1].

    This rank-transformation is useful for visualization, ensuring that
    matrices with vastly different baseline scales can be plotted on the
    same uniform colormap.

    Args:
        rdm (np.ndarray): A symmetric (N, N) NumPy dissimilarity matrix.

    Returns:
        np.ndarray: A symmetric (N, N) matrix where off-diagonal elements
                    are converted to percentiles from 0.0 (most similar)
                    to 1.0 (most dissimilar). The diagonal remains 0.0.
    """
    n = rdm.shape[0]
    i_upper, j_upper = np.triu_indices(n, k=1)

    ranks = rankdata(rdm[i_upper, j_upper])
    percentiles = (ranks - 1) / (len(ranks) - 1)

    percentile_rdm = np.zeros_like(rdm)
    percentile_rdm[i_upper, j_upper] = percentiles
    percentile_rdm[j_upper, i_upper] = percentiles

    return percentile_rdm
