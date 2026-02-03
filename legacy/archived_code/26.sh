#!/bin/bash
#SBATCH -J name
#SBATCH -p week
#SBATCH -c 4
#SBATCH -t 4-01:00:00
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 26_pVariatePyTorch.ipynb 26_pVariatePyTorchOut.ipynb