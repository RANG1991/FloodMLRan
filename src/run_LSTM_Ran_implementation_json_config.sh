#!/bin/bash

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
  python ./FloodML_train_test.py --model CNN_LSTM --json_config_file_name "$json_config_file_name"
  python ./FloodML_train_test.py --model LSTM --json_config_file_name "$json_config_file_name"
done