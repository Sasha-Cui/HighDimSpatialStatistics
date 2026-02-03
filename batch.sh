#!/bin/bash
#SBATCH -p day
#SBATCH -c 1
#SBATCH -t 6:00:00

module purge
module load miniconda
conda activate my_env
papermill /path/to/notebook.ipynb /path/to/output.ipynb

