#!/bin/bash
#SBATCH --job-name=hds_support
#SBATCH --account=pi_jss233
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%A_%a.out
#SBATCH --error=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics
HDS_PYTHON=${HDS_PYTHON:-/nfs/roberts/project/pi_btk22/zc362/environments/current/kt-main/bin/python}
MANIFEST=${1:?"manifest path is required"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO_ROOT"
"$HDS_PYTHON" scripts/research/run_finite_support_study.py \
  --manifest "$MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID"
