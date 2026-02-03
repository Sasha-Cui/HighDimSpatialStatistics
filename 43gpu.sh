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
conda activate myenv

# Use the Slurm job ID and loop index to generate unique filenames
slurm_job_id=${SLURM_JOB_ID}

# Start monitoring VRAM usage every 600 seconds with a unique log file for each run
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 600 > 43_gpu_vram_log_${slurm_job_id}.txt &

# Run Papermill with a unique output filename for each run
papermill 43_fitting_and_validation_metric.ipynb 43_fitting_and_validation_metric.ipynb && \
papermill 44_metric_calculation.ipynb 44_metric_calculation.ipynb

# Stop the monitoring once the job is done
kill %1