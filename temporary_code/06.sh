#!/bin/bash
#SBATCH -p day
#SBATCH -t 00:01:00
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 06.ipynb 06.ipynb