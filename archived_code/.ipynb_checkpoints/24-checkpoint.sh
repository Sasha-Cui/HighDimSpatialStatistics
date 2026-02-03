#!/bin/bash
#SBATCH -J TrivariateScipy
#SBATCH -p bigmem
#SBATCH -c 64
#SBATCH -t 1-00:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  #

module purge
module load miniconda
conda activate myenv
papermill 24_Genton_ThreeDimensional_Corrected.ipynb 24_Genton_ThreeDimensionalOut.ipynb