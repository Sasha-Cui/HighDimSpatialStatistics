#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=479G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv

# Use the Slurm job ID and loop index to generate unique filenames
slurm_job_id=${SLURM_JOB_ID}

papermill 38_MarginalFitting4_3.ipynb 38_MarginalFitting4_3_${slurm_job_id}.ipynb