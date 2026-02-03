"""Linear algebra helpers."""
from __future__ import annotations

import torch


def is_positive_definite(matrix: torch.Tensor) -> bool:
    """Check if a matrix is positive definite using Cholesky."""
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return (matrix + matrix.T) / 2."""
    return (matrix + matrix.mT) / 2
