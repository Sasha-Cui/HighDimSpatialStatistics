## The maximum memory from each partitions follow ##

#SBATCH -p day
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=479G
 

#SBATCH -p week
#SBATCH -t 6-23:59:59
#SBATCH -c 32
#SBATCH --mem=319G


#SBATCH -p bigmem
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=1991G


Usual GPU
#SBATCH -p gpu
#SBATCH -t 0-23:59:59
#SBATCH --qos=qos_ma_zongming
#SBATCH --gres=gpu:h100:1
#SBATCH -c 12
#SBATCH --mem=243G


#SBATCH -p gpu
#SBATCH -t 1-23:59:59
#SBATCH --qos=qos_ma_zongming
#SBATCH --gres=gpu:h100:1
#SBATCH -c 48
#SBATCH --mem=974G


Typical example
#!/bin/bash
#SBATCH -p bigmem
#SBATCH -t 23:59:59
#SBATCH -c 64
#SBATCH --mem=1991G
#SBATCH --mail-type=END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sasha.cui@yale.edu  # Your email address