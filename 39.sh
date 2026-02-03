#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 1-23:59:59
#SBATCH --qos=qos_ma_zongming
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=400G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address

module purge
module load miniconda
conda activate myenv
papermill 39_gpu.ipynb 39_gpu.ipynb