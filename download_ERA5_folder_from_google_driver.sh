#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=3:0:0
#SBATCH --mem=40G

# Uncomment and enter path of code
cd /sci/labs/efratmorin/lab_share/FloodsML/data/ERA5/

# virtual_env location
virtual_env=/sci/labs/efratmorin/lab_share/FloodsMLEnv/bin/activate

source $virtual_env

gdown --folder https://drive.google.com/drive/folders/144ePop3rMeF2TiV-CtVpRoCzgK2ZRGeD?usp=sharing
