#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 10
#SBATCH --time=150:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:1

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

for i in 1 2 3
do
  echo "run number: $i"
  NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model Transformer_CNN --dataset CARAVAN --optim Adam --num_epochs 15 --sequence_length_spatial 7 --num_workers_data_loader 5
done
