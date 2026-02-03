#!/bin/bash
#SBATCH -p bigmem
#SBATCH -t 0:59:59
#SBATCH -c 64
#SBATCH --mem=1991G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 52_kernel_smoothing_real_data.ipynb 52_kernel_smoothing_real_data.ipynb
