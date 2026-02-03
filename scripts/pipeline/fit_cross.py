"""Fit cross-covariance parameters."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts.pipeline._common import add_src_to_path, load_tensors, save_tensors


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.fitting.cross import optimize_cross_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/real_data.pt")
    parser.add_argument("--marginal-params", default="data/processed/marginal_params.csv")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--steps-per-batch", type=int, default=2)
    parser.add_argument("--output", default="data/processed/cross_params.pt")
    parser.add_argument("--use-legacy-kernel", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    args = parser.parse_args()

    payload = load_tensors(repo_root / args.input)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    X = payload["X"].to(torch.float64).to(device)
    Y = payload["Y"].to(torch.float64).to(device)

    df = pd.read_csv(repo_root / args.marginal_params)
    optimized_marginals = [tuple(row) for row in df[["alpha", "nu", "sigma"]].to_numpy()]

    params, best_params, loss_histories = optimize_cross_parameters(
        optimized_marginals,
        X,
        Y,
        number_of_groups=args.groups,
        number_of_cycles=args.cycles,
        steps_per_batch=args.steps_per_batch,
        use_legacy_kernel=args.use_legacy_kernel,
    )

    output_path = repo_root / args.output
    save_tensors(output_path, {"params": params, "best_params": best_params, "loss_histories": loss_histories})
    print(f"Saved cross parameters to {output_path}")


if __name__ == "__main__":
    main()
