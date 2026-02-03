"""DataFrame helpers."""
from __future__ import annotations

import pandas as pd
import torch


def store_as_df(alpha_matrix: torch.Tensor, nu_matrix: torch.Tensor, sigma_matrix: torch.Tensor) -> pd.DataFrame:
    data = {}
    for i in range(alpha_matrix.size(0)):
        for j in range(alpha_matrix.size(1)):
            if i <= j:
                data[f"alpha_matrix_{i+1}{j+1}"] = alpha_matrix[i, j].item()
                data[f"nu_matrix_{i+1}{j+1}"] = nu_matrix[i, j].item()
                data[f"sigma_matrix_{i+1}{j+1}"] = sigma_matrix[i, j].item()
    return pd.DataFrame([data])
