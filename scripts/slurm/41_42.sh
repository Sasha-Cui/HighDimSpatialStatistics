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

N_LOCATIONS="${HDS_N_LOCATIONS:-500}"
SYN_OUT="${HDS_SYN_OUT:-data/synthetic/genton_dataset.pt}"
python -m scripts.pipeline.generate_synthetic --n-locations "$N_LOCATIONS" --output "$SYN_OUT"
SMOOTH_OUT="${HDS_SMOOTH_OUT:-data/processed/kernel_smoothed.pt}"
python -m scripts.pipeline.kernel_smoothing --input "$SYN_OUT" --output "$SMOOTH_OUT"
