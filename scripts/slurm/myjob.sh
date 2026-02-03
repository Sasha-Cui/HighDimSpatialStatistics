#!/bin/bash
#SBATCH -p week
#SBATCH -c 2
#SBATCH -t 96:00:00
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

if [ -z "${HDS_LEGACY_SCRIPT:-}" ]; then
  echo "Set HDS_LEGACY_SCRIPT to a legacy script path (e.g., scripts/legacy/archived_code/23_SubmissionAttempt (nothing here).py)"
  exit 1
fi
python scripts/pipeline/legacy_runner.py --script "$HDS_LEGACY_SCRIPT"
