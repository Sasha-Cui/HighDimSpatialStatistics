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

# Loop to run the Papermill notebook 10 times
for i in {1..10}; do
  # Run Papermill with a unique output filename for each run
  papermill 38_marginal_fitting.ipynb 38_${SLURM_JOB_ID}_${i}.ipynb
done
