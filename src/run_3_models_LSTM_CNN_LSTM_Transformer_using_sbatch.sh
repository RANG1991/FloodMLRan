#! /bin/bash

sbatch --export=json_config_file_name="config_files_json/config_run_local_above_1000_basins_debug.json" ./src/run_CNN_LSTM_sbatch_json_config.sh
sbatch --export=json_config_file_name="config_files_json/config_run_local_above_1000_basins_debug.json" ./src/run_LSTM_sbatch_json_config.sh
sbatch --export=json_config_file_name="config_files_json/config_run_local_above_1000_basins_debug.json" ./src/run_Transformer_sbatch_json_config.sh