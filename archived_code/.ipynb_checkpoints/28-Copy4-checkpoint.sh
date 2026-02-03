#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:59
#SBATCH -c 32
#SBATCH --mem=479G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 28_BatchLearning-Copy4.ipynb 28_BatchLearning-Copy4Out.ipynb