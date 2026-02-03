"""Fit marginal parameters for each feature."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from scripts.pipeline._common import add_src_to_path, load_tensors


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.fitting.marginal import optimize_marginal_parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/real_data.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--steps-per-batch", type=int, default=1)
    parser.add_argument("--output", default="data/processed/marginal_params.csv")
    args = parser.parse_args()

    payload = load_tensors(repo_root / args.input)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    X = payload["X"].to(torch.float64).to(device)
    Y = payload["Y"].to(torch.float64).to(device)

    params = optimize_marginal_parameters(
        X,
        Y,
        number_of_groups=args.groups,
        number_of_cycles=args.cycles,
        steps_per_batch=args.steps_per_batch,
    )

    df = pd.DataFrame(params, columns=["alpha", "nu", "sigma"])
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved marginal parameters to {output_path}")


if __name__ == "__main__":
    main()
