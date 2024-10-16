#!/bin/bash

#SBATCH -n 3
#SBATCH --time=100:0:0
#SBATCH --mem=160G
#SBATCH --gres gpu:3

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
python ./FloodML_runner.py --yaml_config_file_name config_files_yml/config_run_Transformer_CNN.yml
#done