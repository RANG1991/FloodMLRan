#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 11
#SBATCH --time=50:0:0
#SBATCH --mem=40G

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

python ./preprocess_CAMELS_data_non_spatial_and_spatial.py