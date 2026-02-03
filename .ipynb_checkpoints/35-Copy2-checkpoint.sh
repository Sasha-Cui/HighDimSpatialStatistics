#!/bin/bash
#SBATCH -p day
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 35_HattieMultipleData-LocGeneTradeoff.ipynb 35_HattieMultipleData-LocGeneTradeoffOut.ipynb