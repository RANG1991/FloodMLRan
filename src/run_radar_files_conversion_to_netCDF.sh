#!/bin/bash

#SBATCH -n 2
#SBATCH --time=100:0:0
#SBATCH --mem=50G

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/src/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env

python ./convert_radar_data_to_netCDF.py