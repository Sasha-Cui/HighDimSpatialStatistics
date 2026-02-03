#!/bin/bash
#SBATCH -p week
#SBATCH -c 2
#SBATCH -t 96:00:00
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sashacui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 23_SubmissionAttempt.ipynb 23_SubmissionAttemptOut.ipynb