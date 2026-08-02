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

if [ -z "${HDS_GENES:-}" ]; then
  echo "Set HDS_GENES to a comma-separated gene list"
  exit 1
fi
REAL_FILE="${HDS_REAL_FILE:-ovary_Puck_230517_39.h5ad}"
REAL_OUT="${HDS_REAL_OUT:-data/processed/real_data.pt}"
python -m scripts.pipeline.preprocess_real --filename "$REAL_FILE" --subdir raw --genes "$HDS_GENES" --output "$REAL_OUT"
SMOOTH_OUT="${HDS_SMOOTH_OUT:-data/processed/kernel_smoothed_real.pt}"
python -m scripts.pipeline.kernel_smoothing --input "$REAL_OUT" --output "$SMOOTH_OUT"
