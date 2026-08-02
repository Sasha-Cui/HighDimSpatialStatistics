#!/bin/bash
#SBATCH -p day
#SBATCH -c 1
#SBATCH -t 6:00:00

module purge
module load miniconda
conda activate "${HDS_CONDA_ENV:-research}"

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
LOG_DIR="${HDS_LOG_DIR:-data/processed/logs}"
mkdir -p "$LOG_DIR"

if [ -z "${HDS_LEGACY_SCRIPT:-}" ]; then
  echo "Set HDS_LEGACY_SCRIPT to a legacy script path (e.g., scripts/legacy/33_DirectContestwR.py)"
  exit 1
fi
python -m scripts.pipeline.legacy_runner --script "$HDS_LEGACY_SCRIPT"
