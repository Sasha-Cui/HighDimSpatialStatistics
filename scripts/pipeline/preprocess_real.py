"""Load and preprocess real spatial data from a .h5ad file."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.pipeline._common import add_src_to_path, save_tensors


def parse_genes(arg: str):
    return [g.strip() for g in arg.split(",") if g.strip()]


def main() -> None:
    repo_root = add_src_to_path()
    from HighDimSpatial.data.real import load_real_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="ovary_Puck_230517_39.h5ad")
    parser.add_argument("--subdir", default="raw")
    parser.add_argument("--genes", required=True, help="Comma-separated gene list")
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--output", default="data/processed/real_data.pt")
    args = parser.parse_args()

    gene_list = parse_genes(args.genes)
    result = load_real_data(
        gene_list=gene_list,
        filename=args.filename,
        subdir=args.subdir,
        head=args.head,
        puck_list=None,
        prefer_cuda=False,
    )

    output_path = repo_root / args.output
    save_tensors(
        output_path,
        {"X": result.X.cpu(), "Y": result.Y.cpu(), "gene_list": result.gene_list},
    )
    print(f"Saved preprocessed real data to {output_path}")
    if result.dropped_genes:
        print(f"Dropped genes: {result.dropped_genes}")


if __name__ == "__main__":
    main()
