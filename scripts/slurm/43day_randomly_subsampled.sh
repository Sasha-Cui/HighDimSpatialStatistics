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


# Run 43.ipynb and only proceed to 44.ipynb if it completes successfully
python scripts/pipeline/legacy_runner.py --script scripts/legacy/43_randomly_subsampled_fitting_and_validation_metric.py
python scripts/pipeline/legacy_runner.py --script scripts/legacy/44_randomply_subsampled_metric_calculation.py
