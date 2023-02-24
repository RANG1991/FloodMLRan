#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs


#SBATCH -n 2
#SBATCH --time=10:0:0
#SBATCH --mem=120G
#SBATCH --gres=gpu:a30:1

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
#module load cuda/11.7

# Start Running NeuralHydrology code of Frederik
MPLBACKEND='Agg' /sci/labs/efratmorin/ranga/FloodMLRan/neuralhydrology/nh_run.py train --config-file /sci/labs/efratmorin/ranga/FloodMLRan/config_files_dir/config_CAMELS.yml