#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=50:0:0
#SBATCH --mem=40G
#SBATCH --gres gpu
#SBATCH -J jupyter

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/FloodsMLEnv/bin/activate

source $virtual_env
# module load cuda/11.2

python ./src/convert_era_5_to_seperate_basins.py