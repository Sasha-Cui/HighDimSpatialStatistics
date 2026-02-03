#!/bin/bash
#SBATCH -p week
#SBATCH -t 6-23:59:59
#SBATCH -c 32
#SBATCH --mem=319G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

# Loop to run the Papermill notebook 10 times
for i in {1..10}; do
  module purge
  module load miniconda
  conda activate myenv

  # Use the Slurm job ID and loop index to generate unique filenames
  slurm_job_id=${SLURM_JOB_ID}

  # Run Papermill with a unique output filename for each run
  papermill 39_cpu.ipynb 39_cpu_${i}_${slurm_job_id}.ipynb

done
