#!/bin/bash
#SBATCH -p week
#SBATCH -t 6-23:59:59
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 33_DirectContestwR.ipynb 33_DirectContestwROut.ipynb