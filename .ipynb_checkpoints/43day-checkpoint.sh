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

# Run 43.ipynb and only proceed to 44.ipynb if it completes successfully
papermill 43_fitting_and_validation_metric.ipynb 43_fitting_and_validation_metric.ipynb