#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=3:0:0
#SBATCH --mem=4G
#SBATCH -J jupyter

## Get host IP
host_ip=$(hostname -i)

# Select free random port
jupyter_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');

# Print tunnel command
echo -e "Open a new terminal window in Mac and Linux or a local terminal in Mobaxterm\n"
echo -e "Copy and paste the command bellow in new terminal to create a tunnel to the cluster\n"
echo -e "ssh -J $USER@bava.cs.huji.ac.il -L $jupyter_port:$host_ip:$jupyter_port $USER@moriah-gw.cs.huji.ac.il\n"
echo -e "After creating the tunnel copy and paste the URL that Jupyter Notebook created\n"

# Uncomment and enter path of code
cd /sci/labs/efratmorin/ranga/CNN_LSTM_v3.ipynb

# Start Jupyter
jupyter-notebook --no-browser --ip=0.0.0.0 --port=$jupyter_port