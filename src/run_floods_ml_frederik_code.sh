#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=10:0:0
#SBATCH --mem=40G
#SBATCH --gres gpu
#SBATCH -J jupyter

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodsML/

# virtual_env location
virtual_env=/sci/labs/efratmorin/lab_share/FloodsMLEnv/bin/activate

source $virtual_env
# module load cuda/11.2

# Start Running NeuralHydrology code of Frederik
./neuralhydrology/neuralhydrology/nh_run_scheduler.py train --directory /sci/labs/efratmorin/ranga/FloodsML/config_files_dir --runs-per-gpu 1 --gpu-ids 0
