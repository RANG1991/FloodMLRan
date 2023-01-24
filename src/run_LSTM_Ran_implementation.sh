#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 3
#SBATCH --time=50:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:3

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

python ./FloodML_train_test.py --model LSTM --dataset CARAVAN --optim Adam --shared_model True --num_epochs 50
