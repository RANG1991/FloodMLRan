#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 5
#SBATCH --time=150:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:1

# --gres gpu:a100-3-40

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

#for i in 1 2 3
#do
#  echo "run number: $i"
python ./FloodML_runner.py --model LSTM --dataset CAMELS --optim Adam --num_epochs 10 --sequence_length_spatial 14 --num_processes_ddp 3 --limit_size_above_1000 --num_workers_data_loader 2 --batch_size 1024
#done
