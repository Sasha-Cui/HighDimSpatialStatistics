#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1-23:59:59
#SBATCH --qos=qos_ma_zongming
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=100G
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


# Loop to run the Papermill notebook 10 times
for i in {1..10}; do

  # Start monitoring VRAM usage every 600 seconds with a unique log file for each run
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 600 > ${LOG_DIR}/39_gpu_log_${SLURM_JOB_ID}_${i}.txt &

  # Run Papermill with a unique output filename for each run
python scripts/pipeline/legacy_runner.py --script scripts/legacy/39_gpu.py

  # Stop the monitoring once the job is done
  kill %1
done
