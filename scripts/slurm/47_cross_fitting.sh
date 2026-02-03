#!/bin/bash
#SBATCH -p week
#SBATCH -t 6-23:59:59
#SBATCH -c 32
#SBATCH --mem=319G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate "${HDS_CONDA_ENV:-research}"

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
LOG_DIR="${HDS_LOG_DIR:-data/processed/logs}"
mkdir -p "$LOG_DIR"

DEVICE="${HDS_DEVICE:-cpu}"
INPUT="${HDS_INPUT:-data/synthetic/genton_dataset.pt}"
MARGINAL_OUT="${HDS_MARGINAL_OUT:-data/processed/marginal_params.csv}"
CROSS_OUT="${HDS_CROSS_OUT:-data/processed/cross_params.pt}"
python scripts/pipeline/fit_cross.py --input "$INPUT" --marginal-params "$MARGINAL_OUT" --output "$CROSS_OUT" --device "$DEVICE"
