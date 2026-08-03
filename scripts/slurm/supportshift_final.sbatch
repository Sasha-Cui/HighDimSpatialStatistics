#!/bin/bash
#SBATCH --job-name=supportshift_final
#SBATCH --account=pi_jss233
#SBATCH --partition=day
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%j.out
#SBATCH --error=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics/outputs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT=/nfs/roberts/project/pi_btk22/zc362/HighDimSpatialStatistics
HDS_PYTHON=${HDS_PYTHON:-/nfs/roberts/project/pi_btk22/zc362/environments/current/kt-main/bin/python}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$REPO_ROOT"
mkdir -p outputs/slurm outputs/smoothing_bias

"$HDS_PYTHON" scripts/research/run_anisotropic_phase_oracle.py \
  --output outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv

"$HDS_PYTHON" scripts/research/run_supportshift_highdim.py \
  --preset full \
  --output outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv \
  --raw-example-output outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv

"$HDS_PYTHON" scripts/research/make_support_paper_artifacts.py \
  --phase paper/data/phase_oracle_d2.csv \
  --finite-summary paper/data/finite_summary.csv \
  --anisotropy outputs/smoothing_bias/supportshift_anisotropic_final_20260803.csv \
  --highdim outputs/smoothing_bias/supportshift_highdim_final_v2_20260803.csv \
  --raw-example outputs/smoothing_bias/supportshift_raw_final_v2_20260803.csv \
  --paper-directory paper
