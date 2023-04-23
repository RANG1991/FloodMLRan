#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 8
#SBATCH --time=150:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:3

# --gres gpu:a100-3-40

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

for i in 1 2 3
do
  echo "run number: $i"
  python ./FloodML_train_test.py --model CNN_LSTM --dataset CAMELS --optim Adam --num_epochs 1 --sequence_length_spatial 14 --num_processes_ddp 3 --limit_size_above_1000 --num_workers_data_loader 2 --batch_size 512 --create_new_files
  python ./FloodML_train_test.py --model LSTM --dataset CAMELS --optim Adam --num_epochs 1 --sequence_length_spatial 14 --num_processes_ddp 3 --limit_size_above_1000 --num_workers_data_loader 2 --batch_size 512 --create_new_files
#  NCCL_P2P_DISABLE=1 python ./FloodML_train_test.py --model CONV_LSTM --dataset CARAVAN --optim Adam --num_epochs 15 --sequence_length_spatial 7 --limit_size_above_1000
done
