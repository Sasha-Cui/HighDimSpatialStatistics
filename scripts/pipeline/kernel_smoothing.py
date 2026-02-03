"""Kernel smoothing for spatial data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.pipeline._common import add_src_to_path, load_tensors, save_tensors


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.data.smoothing import kernel_smoothing

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/real_data.pt")
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--number-of-grids", type=int, default=10)
    parser.add_argument("--min-grid-count", type=int, default=100)
    parser.add_argument("--max-grid-count", type=int, default=5000)
    parser.add_argument("--output", default="data/processed/kernel_smoothed.pt")
    args = parser.parse_args()

    input_path = repo_root / args.input
    payload = load_tensors(input_path)
    X = payload["X"].to(torch.float64)
    Y = payload["Y"].to(torch.float64)

    X_groups, Y_groups = kernel_smoothing(
        X,
        Y,
        bandwidth=torch.tensor(args.bandwidth, dtype=torch.float64),
        number_of_grids=args.number_of_grids,
        min_grid_count=args.min_grid_count,
        max_grid_count=args.max_grid_count,
    )

    output_path = repo_root / args.output
    save_tensors(output_path, {"X_groups": X_groups, "Y_groups": Y_groups})
    print(f"Saved kernel-smoothed data to {output_path}")


if __name__ == "__main__":
    main()
