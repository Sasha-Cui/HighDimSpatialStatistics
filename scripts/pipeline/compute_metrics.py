"""Compute validation metrics for marginal parameters."""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from scripts.pipeline._common import add_src_to_path, load_tensors


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.metrics.validation import (
        extract_location_major_marginal_covariance,
        validation_metric_marginal,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/genton_dataset.pt")
    parser.add_argument("--marginal-params", default="data/processed/marginal_params.csv")
    parser.add_argument("--output", default="data/processed/validation_metrics.csv")
    args = parser.parse_args()

    payload = load_tensors(repo_root / args.input)
    X = payload["X"].to(torch.float64)
    K = payload.get("K")
    if K is None:
        raise ValueError("Input payload must include 'K' (true covariance) to compute metrics.")

    df = pd.read_csv(repo_root / args.marginal_params)
    params = [
        {"alpha": torch.tensor(row.alpha, dtype=torch.float64),
         "nu": torch.tensor(row.nu, dtype=torch.float64),
         "sigma": torch.tensor(row.sigma, dtype=torch.float64)}
        for row in df.itertuples(index=False)
    ]

    p = len(params)
    if K.shape != (X.size(0) * p, X.size(0) * p):
        raise ValueError(
            "K shape is inconsistent with the number of locations and marginal parameter rows"
        )
    metrics = []
    for i, param in enumerate(params):
        K_test = extract_location_major_marginal_covariance(K, i, p)
        metric = validation_metric_marginal(param, X, K_test)
        metrics.append({"feature": i, "metric": metric})

    out_df = pd.DataFrame(metrics)
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved validation metrics to {out_path}")


if __name__ == "__main__":
    main()
