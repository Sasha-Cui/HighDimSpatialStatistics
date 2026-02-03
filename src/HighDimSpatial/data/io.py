"""I/O helpers for spatial transcriptomics datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import scanpy as sc
import torch

from HighDimSpatial.config import DEFAULT_DTYPE, resolve_data_path


def read_h5ad(path: Path) -> sc.AnnData:
    """Read a .h5ad file from a path."""
    return sc.read_h5ad(str(path))


def load_h5ad(
    filename: str,
    subdir: Optional[str] = None,
) -> sc.AnnData:
    """Load a .h5ad file from the data directory.

    Args:
        filename: File name under data dir.
        subdir: Optional subdir under data dir.
    """
    if subdir:
        path = resolve_data_path(subdir, filename, must_exist=True)
    else:
        path = resolve_data_path(filename, must_exist=True)
    return read_h5ad(path)


def extract_gene_values(
    adata: sc.AnnData,
    gene_name: str,
    device: torch.device,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Extract gene expression values for one gene as a tensor.

    Prefers `adata.raw` if present.
    """
    if adata.raw is not None:
        gene_data = adata.raw[:, gene_name].X
    else:
        gene_data = adata[:, gene_name].X

    # convert to dense
    if hasattr(gene_data, "toarray"):
        gene_data = gene_data.toarray()

    return torch.tensor(gene_data, dtype=dtype, device=device)


def extract_spatial_coordinates(
    adata: sc.AnnData,
    key: str = "spatial",
    device: torch.device | None = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Extract spatial coordinates from `adata.obsm[key]`."""
    coords = adata.obsm[key]
    return torch.tensor(coords, dtype=dtype, device=device)
