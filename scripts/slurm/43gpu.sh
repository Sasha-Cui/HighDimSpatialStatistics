#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1-23:59:59
#SBATCH --qos=qos_ma_zongming
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=150G
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

DEVICE="${HDS_DEVICE:-cuda}"
INPUT="${HDS_INPUT:-data/synthetic/genton_dataset.pt}"
MARGINAL_OUT="${HDS_MARGINAL_OUT:-data/processed/marginal_params.csv}"
python -m scripts.pipeline.fit_marginals --input "$INPUT" --output "$MARGINAL_OUT" --device "$DEVICE"
METRICS_OUT="${HDS_METRICS_OUT:-data/processed/validation_metrics.csv}"
python -m scripts.pipeline.compute_metrics --input "$INPUT" --marginal-params "$MARGINAL_OUT" --output "$METRICS_OUT"
