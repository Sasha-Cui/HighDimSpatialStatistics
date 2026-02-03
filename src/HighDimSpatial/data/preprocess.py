"""Preprocessing helpers for spatial expression data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from HighDimSpatial.config import DEFAULT_DTYPE


@dataclass
class SpatialPreprocessConfig:
    scale_divisor: float = 5000.0
    x_offset: float = 0.5948
    y_offset: float = 0.5389
    log1p: bool = True
    normalize_gene_sums: bool = True
    normalize_gene_std: bool = True


@dataclass
class PreprocessResult:
    X: torch.Tensor
    Y: torch.Tensor
    gene_list: List[str]
    dropped_genes: List[str]


def filter_zero_sum_genes(
    gene_data: torch.Tensor,
    gene_list: Sequence[str],
) -> tuple[torch.Tensor, list[str], list[str], torch.Tensor]:
    """Drop genes whose expression sums to zero.

    Returns filtered data, filtered gene list, dropped gene list, and column sums.
    """
    column_sums = torch.sum(gene_data, dim=0)
    non_zero_mask = column_sums != 0
    dropped = [gene_list[i] for i in range(len(gene_list)) if not non_zero_mask[i].item()]
    filtered = gene_data[:, non_zero_mask]
    filtered_sums = column_sums[non_zero_mask]
    filtered_genes = [gene_list[i] for i in range(len(gene_list)) if non_zero_mask[i].item()]
    return filtered, filtered_genes, dropped, filtered_sums


def preprocess_spatial_data(
    coordinates: torch.Tensor,
    gene_data: torch.Tensor,
    gene_list: Sequence[str],
    config: SpatialPreprocessConfig | None = None,
) -> PreprocessResult:
    """Apply standard preprocessing used in the original notebooks.

    Steps:
    - Drop zero-sum genes and optionally normalize by column sums
    - Scale x/y coordinates
    - Log1p transform gene data
    - Normalize gene data by column std
    """
    if config is None:
        config = SpatialPreprocessConfig()

    dropped_genes: list[str] = []

    if config.normalize_gene_sums:
        gene_data, gene_list, dropped_genes, column_sums = filter_zero_sum_genes(gene_data, gene_list)
        gene_data = gene_data / column_sums

    # Scale coordinates
    coordinates = coordinates.clone()
    coordinates[:, 0] = coordinates[:, 0] / config.scale_divisor - config.x_offset
    coordinates[:, 1] = coordinates[:, 1] / config.scale_divisor - config.y_offset

    if config.log1p:
        gene_data = torch.log1p(gene_data)

    if config.normalize_gene_std:
        std_devs = torch.std(gene_data, dim=0)
        if torch.any(std_devs == 0):
            # Avoid divide-by-zero by leaving those columns unchanged
            mask = std_devs != 0
            gene_data[:, mask] = gene_data[:, mask] / std_devs[mask]
        else:
            gene_data = gene_data / std_devs

    return PreprocessResult(X=coordinates, Y=gene_data, gene_list=list(gene_list), dropped_genes=dropped_genes)


def shuffle_and_subset(
    coordinates: torch.Tensor,
    gene_data: torch.Tensor,
    head: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle rows and optionally subset to the first `head` rows."""
    n = gene_data.size(0)
    indices = torch.randperm(n, device=gene_data.device)
    if head:
        indices = indices[:head]
    return coordinates[indices], gene_data[indices]
