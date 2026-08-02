"""Seeded pilot study of covariance inference after linear spatial smoothing.

This script deliberately compares two likelihoods for the same smoothed data:

1. corrected: ``S K_theta S.T``;
2. naive: ``K_theta`` evaluated at the output locations.

It is a research scaffold, not evidence for a paper until the preregistered
simulation grid in ``docs/research/SIMULATION_PROTOCOL.md`` has been run.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch


def add_src_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    return repo_root


def smoothing_operator(X: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Construct a row-normalized Epanechnikov smoother on observed locations."""
    distances = torch.cdist(X, X)
    scaled = distances / bandwidth
    weights = 0.75 * (1.0 - scaled.square()) * (scaled < 1).to(X.dtype)
    return weights / weights.sum(dim=1, keepdim=True)


def evaluate_alpha_grid(
    y_smoothed: torch.Tensor,
    X: torch.Tensor,
    S: torch.Tensor,
    alpha_grid: torch.Tensor,
    nu: float,
    sigma: float,
    nugget: float,
) -> tuple[float, float, float, float]:
    from HighDimSpatial.data.smoothing import transform_location_covariance
    from HighDimSpatial.kernels.matern import matern_kernel
    from HighDimSpatial.metrics.likelihood import negative_log_likelihood

    distances = torch.cdist(X, X)
    corrected_losses: list[float] = []
    naive_losses: list[float] = []
    identity = torch.eye(X.size(0), dtype=X.dtype)
    for alpha in alpha_grid:
        latent_covariance = matern_kernel(
            distances,
            alpha,
            torch.tensor(nu, dtype=X.dtype),
            torch.tensor(sigma, dtype=X.dtype),
        )
        observed_covariance = latent_covariance + nugget * identity
        corrected_covariance = transform_location_covariance(
            observed_covariance, S, number_of_variables=1
        )
        # A tiny numerical jitter is separate from the generative nugget.
        corrected_covariance += 1e-10 * identity
        naive_covariance = latent_covariance + nugget * identity
        corrected_losses.append(
            float(negative_log_likelihood(y_smoothed, corrected_covariance).item())
        )
        naive_losses.append(float(negative_log_likelihood(y_smoothed, naive_covariance).item()))

    corrected_index = int(np.argmin(corrected_losses))
    naive_index = int(np.argmin(naive_losses))
    return (
        float(alpha_grid[corrected_index].item()),
        float(alpha_grid[naive_index].item()),
        corrected_losses[corrected_index],
        naive_losses[naive_index],
    )


def run_replicate(
    n_locations: int,
    domain_length: float,
    bandwidth: float,
    alpha_true: float,
    alpha_grid: torch.Tensor,
    nu: float,
    sigma: float,
    nugget: float,
    seed: int,
) -> dict[str, float | int]:
    from HighDimSpatial.kernels.matern import matern_kernel
    from HighDimSpatial.utils.torch_utils import set_seed

    set_seed(seed)
    X = torch.linspace(0.0, domain_length, n_locations, dtype=torch.float64).reshape(-1, 1)
    distances = torch.cdist(X, X)
    covariance = matern_kernel(
        distances,
        torch.tensor(alpha_true, dtype=torch.float64),
        torch.tensor(nu, dtype=torch.float64),
        torch.tensor(sigma, dtype=torch.float64),
    )
    covariance += nugget * torch.eye(n_locations, dtype=torch.float64)
    y = torch.distributions.MultivariateNormal(
        torch.zeros(n_locations, dtype=torch.float64), covariance_matrix=covariance
    ).sample()
    S = smoothing_operator(X, bandwidth)
    y_smoothed = S @ y
    alpha_corrected, alpha_naive, nll_corrected, nll_naive = evaluate_alpha_grid(
        y_smoothed, X, S, alpha_grid, nu, sigma, nugget
    )
    return {
        "seed": seed,
        "n_locations": n_locations,
        "domain_length": domain_length,
        "bandwidth": bandwidth,
        "alpha_true": alpha_true,
        "alpha_corrected": alpha_corrected,
        "alpha_naive": alpha_naive,
        "absolute_error_corrected": abs(alpha_corrected - alpha_true),
        "absolute_error_naive": abs(alpha_naive - alpha_true),
        "minimum_nll_corrected": nll_corrected,
        "minimum_nll_naive": nll_naive,
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    repo_root = add_src_to_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-locations", type=int, default=40)
    parser.add_argument("--domain-length", type=float, default=4.0)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--alpha-true", type=float, default=1.0)
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=2.5)
    parser.add_argument("--alpha-grid-size", type=int, default=30)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--nugget", type=float, default=0.05)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "outputs" / "smoothing_bias" / "pilot.jsonl",
    )
    args = parser.parse_args()
    alpha_grid = torch.linspace(
        args.alpha_min, args.alpha_max, args.alpha_grid_size, dtype=torch.float64
    )

    metadata = {
        "record_type": "metadata",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "arguments": vars(args) | {"output": str(args.output)},
    }
    append_jsonl(args.output, metadata)
    for replicate in range(args.replicates):
        result = run_replicate(
            args.n_locations,
            args.domain_length,
            args.bandwidth,
            args.alpha_true,
            alpha_grid,
            args.nu,
            args.sigma,
            args.nugget,
            args.seed + replicate,
        )
        append_jsonl(args.output, {"record_type": "replicate", **result})
    print(f"Wrote {args.replicates} replicates to {args.output}")


if __name__ == "__main__":
    main()
