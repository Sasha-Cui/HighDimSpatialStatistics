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

papermill 47_cross_fitting.ipynb 47_cross_fitting.ipynb