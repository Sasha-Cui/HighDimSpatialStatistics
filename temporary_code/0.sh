#!/bin/bash
#SBATCH -p bigmem
#SBATCH -c 64
#SBATCH -t 1-00:00:00
#SBATCH --mem-per-cpu=31G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 01_importing_util.ipynb 01_out.ipynb