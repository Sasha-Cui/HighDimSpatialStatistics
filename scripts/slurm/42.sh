#!/bin/bash
#SBATCH -p bigmem
#SBATCH -t 0:59:59
#SBATCH -c 64
#SBATCH --mem=1991G
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

INPUT="${HDS_INPUT:-data/synthetic/genton_dataset.pt}"
OUTPUT="${HDS_OUTPUT:-data/processed/kernel_smoothed.pt}"
python scripts/pipeline/kernel_smoothing.py --input "$INPUT" --output "$OUTPUT"
