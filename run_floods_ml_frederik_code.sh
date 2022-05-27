#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=3:0:0
#SBATCH --mem=40G
#SBATCH --gres gpu
#SBATCH -J jupyter

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/FloodsMLEnv/bin/activate

source $virtual_env
# module load cuda/11.2

# Start Running NeuralHydrology code of Frederik
/sci/labs/efratmorin/ranga/FloodMLRan/neuralhydrology/nh_run.py train --config-file /sci/labs/efratmorin/ranga/FloodMLRan/config_files_dir/config.yml --runs-per-gpu 1 --gpu-ids 0
