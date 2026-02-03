#!/bin/bash
#SBATCH -p week
#SBATCH -t 2-23:59:59
#SBATCH -c 32
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address
module purge
module load miniconda
conda activate myenv
papermill 28_BatchLearning-Copy2.ipynb 28_BatchLearning-Copy2Out.ipynb