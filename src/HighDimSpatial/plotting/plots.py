"""Plotting helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_gp_data(X: torch.Tensor, Y: torch.Tensor) -> None:
    """Plot p-variate data for each variable."""
    p = Y.size(1)
    for i in range(p):
        plt.figure(figsize=(8, 6))
        if X.size(1) == 2:
            plt.scatter(X[:, 0].detach().cpu().numpy(), X[:, 1].detach().cpu().numpy(), c=Y[:, i].detach().cpu().numpy(), cmap="viridis", s=1)
            plt.colorbar(label=f"Variable {i+1}")
            plt.xlabel("X1")
            plt.ylabel("X2")
        elif X.size(1) == 1:
            plt.plot(X.detach().cpu().numpy(), Y[:, i].detach().cpu().numpy(), "-o")
            plt.xlabel("X")
            plt.ylabel(f"Variable {i+1}")

        plt.title(f"Visualization of Variable {i+1}")
        plt.tight_layout()
        plt.show()
