#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 3
#SBATCH --time=50:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:1

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

for i in 1 2 3 4 5
do
  echo "run number: $i"
  NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model CNN_LSTM --dataset CARAVAN --optim Adam --num_epochs 10 --sequence_length_spatial 7 --limit_size_above_1000 --num_workers_data_loader 2 --create_new_files --use_all_static_attr
  NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model LSTM --dataset CARAVAN --optim Adam --num_epochs 10 --sequence_length_spatial 7 --limit_size_above_1000 --num_workers_data_loader 2 --create_new_files --use_all_static_attr
#  NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model CONV_LSTM --dataset CARAVAN --optim Adam --num_epochs 15 --sequence_length_spatial 7 --limit_size_above_1000
done
