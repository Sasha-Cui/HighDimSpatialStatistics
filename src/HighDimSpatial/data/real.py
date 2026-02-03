"""Real data loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import scanpy as sc
import torch

from HighDimSpatial.config import DEFAULT_DTYPE, get_device, resolve_data_path
from HighDimSpatial.data.io import extract_gene_values, extract_spatial_coordinates, read_h5ad
from HighDimSpatial.data.preprocess import SpatialPreprocessConfig, preprocess_spatial_data, shuffle_and_subset


@dataclass
class RealDataLoadResult:
    adata: sc.AnnData
    X: torch.Tensor
    Y: torch.Tensor
    gene_list: List[str]
    dropped_genes: List[str]


def load_real_data(
    gene_list: Sequence[str],
    filename: str,
    subdir: Optional[str] = "raw",
    head: int = 0,
    puck_list: Optional[Sequence[str]] = None,
    prefer_cuda: bool = True,
    preprocess_config: SpatialPreprocessConfig | None = None,
) -> RealDataLoadResult:
    """Load and preprocess real spatial data from a .h5ad file.

    Args:
        gene_list: Genes to extract.
        filename: .h5ad filename under data directory (or subdir).
        subdir: Subdirectory under data. Default "raw".
        head: Optional number of rows to sample (after shuffle).
        puck_list: Optional list of pucks to filter.
        prefer_cuda: Use GPU if available.
        preprocess_config: Optional preprocessing configuration.
    """
    device = get_device(prefer_cuda)
    data_path = resolve_data_path(subdir, filename, must_exist=True) if subdir else resolve_data_path(filename, must_exist=True)
    adata = read_h5ad(data_path)

    if puck_list is not None:
        adata = adata[adata.obs["puck"].isin(puck_list)]

    coordinates = extract_spatial_coordinates(adata, device=device, dtype=DEFAULT_DTYPE)
    gene_tensors = [extract_gene_values(adata, gene, device=device, dtype=DEFAULT_DTYPE) for gene in gene_list]
    gene_data = torch.cat(gene_tensors, dim=1)

    coordinates, gene_data = shuffle_and_subset(coordinates, gene_data, head=head)
    result = preprocess_spatial_data(coordinates, gene_data, gene_list, preprocess_config)

    return RealDataLoadResult(
        adata=adata,
        X=result.X,
        Y=result.Y,
        gene_list=result.gene_list,
        dropped_genes=result.dropped_genes,
    )
