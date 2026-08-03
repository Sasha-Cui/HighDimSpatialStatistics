#!/bin/bash
#SBATCH --job-name=hds_reduce
#SBATCH --account=pi_jss233
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics
HDS_PYTHON=${HDS_PYTHON:-/nfs/roberts/project/pi_btk22/zc362/environments/current/kt-main/bin/python}
MANIFEST=${1:?"manifest path is required"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO_ROOT"
"$HDS_PYTHON" scripts/research/reduce_finite_support_study.py \
  --manifest "$MANIFEST" \
  --output-root "$REPO_ROOT/outputs/smoothing_bias"
