"""Effective rank of a node-representation matrix (Roy & Vetterli, 2007).

Used as the over-smoothing measure recommended by Zhang et al. (ICLR 2026,
arXiv:2502.04591): it collapses towards 1 when all node representations
become collinear, where the Dirichlet energy stays uninformative for shallow
networks.
"""
from __future__ import annotations

import math

import torch


def effective_rank(features: torch.Tensor) -> float:
    """Exponential of the Shannon entropy of the normalized singular values.

    Args:
        features: matrix of shape ``(num_nodes, num_channels)``.

    Returns:
        A value in ``[1, min(num_nodes, num_channels)]``, or ``nan`` when the
        matrix is identically zero.
    """
    singular_values = torch.linalg.svdvals(features.detach().double())
    total = singular_values.sum()
    if total <= 0:
        return math.nan
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return float(torch.exp(entropy))
