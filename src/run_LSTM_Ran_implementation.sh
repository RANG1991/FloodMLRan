#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=50:0:0
#SBATCH --mem=40G
#SBATCH --gres gpu

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

python ./FloodML_train_test.py --model LSTM --dataset CARAVAN --optim Adam --shared_model True
