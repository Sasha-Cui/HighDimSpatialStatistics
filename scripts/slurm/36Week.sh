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

python scripts/pipeline/legacy_runner.py --script scripts/legacy/archived_code/36_SubsamplingwReplacementWeek.py
