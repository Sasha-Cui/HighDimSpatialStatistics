"""Generate Genton synthetic data and save tensors to disk."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.pipeline._common import add_src_to_path, save_tensors


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.data.synthetic import generate_genton_synthetic
    from HighDimSpatial.utils.torch_utils import set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-locations", type=int, default=500)
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/synthetic/genton_dataset.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    result = generate_genton_synthetic(args.n_locations, args.dims)
    output_path = repo_root / args.output
    save_tensors(
        output_path,
        {
            "X": result.X,
            "Y": result.Y,
            "K": result.K,
            "alpha_matrix": result.alpha_matrix,
            "nu_matrix": result.nu_matrix,
            "sigma_matrix": result.sigma_matrix,
        },
    )
    print(f"Saved synthetic dataset to {output_path}")


if __name__ == "__main__":
    main()
