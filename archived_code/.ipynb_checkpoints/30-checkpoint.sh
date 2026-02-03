#!/bin/bash
#SBATCH -J my_r_program
#SBATCH -p bigmem
#SBATCH -c 64
#SBATCH -t 1-00:00:00
#SBATCH --mem=1991G
#SBATCH --mail-type=ALL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  #
module load R/4.4.1-foss-2022b
Rscript ~/project/GpGp_multi_paper/R_scripts/install_packages.R
Rscript 30_JoeGuinnessScript.R