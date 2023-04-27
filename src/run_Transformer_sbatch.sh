#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 5
#SBATCH --time=150:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:a30:1

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env
# module load cuda/11.2

#for i in 1 2 3
#do
#  echo "run number: $i"
python ./FloodML_train_test.py --model Transformer_Encoder --dataset CAMELS --optim Adam --num_epochs 10 --sequence_length_spatial 7 --num_workers_data_loader 3 --use_all_static_attr --create_new_files --save_checkpoint
#done
