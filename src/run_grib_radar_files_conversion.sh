#!/bin/bash

#SBATCH -n 2
#SBATCH --time=100:0:0
#SBATCH --mem=160G

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/FloodMLRan/data/data.eol.ucar.edu/pub/download/extra/katz_data/stage4/

# virtual_env location
virtual_env=/sci/labs/efratmorin/ranga/PythonEnvFloodML/bin/activate

source $virtual_env

find . -name '*.Z' -type f -exec /sci/labs/efratmorin/ranga/local/wgrib/wgrib {} -text \;