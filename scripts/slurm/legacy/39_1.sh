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
conda activate myenv

# Loop to run the Papermill notebook 10 times
for i in {1..10}; do

  # Start monitoring VRAM usage every 600 seconds with a unique log file for each run
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 600 > 39_gpu_log_${SLURM_JOB_ID}_${i}.txt &

  # Run Papermill with a unique output filename for each run
  papermill 39_gpu.ipynb 39_gpu_${SLURM_JOB_ID}_${i}.ipynb

  # Stop the monitoring once the job is done
  kill %1
done
