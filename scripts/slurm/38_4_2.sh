#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=479G
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


# Use the Slurm job ID and loop index to generate unique filenames
slurm_job_id=${SLURM_JOB_ID}

python -m scripts.pipeline.legacy_runner --script scripts/legacy/archived_code/38_MarginalFitting4_2.py
