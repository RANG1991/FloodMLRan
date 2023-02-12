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

NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model CNN_LSTM --dataset CARAVAN --optim Adam --shared_model True --num_epochs 10 --sequence_length_spatial 7
NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model LSTM --dataset CARAVAN --optim Adam --shared_model True --num_epochs 10 --sequence_length_spatial 7