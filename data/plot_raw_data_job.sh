#!/bin/bash
#BSUB -J plot_raw_data
#BSUB -q hpc                # Use CPU queue (as GPUs are not needed at all, since it is only for loading and plotting). 
#BSUB -W 1:00               # 1 hour 
#BSUB -n 1                  # 1 Core
#BSUB -R "rusage[mem=64GB]" # Request 64GB RAM to handle the big tensors with the expression data. 
#BSUB -R "span[hosts=1]"
#BSUB -o Misc/results/plot_tensor_%J.out
#BSUB -e Misc/results/plot_tensor_%J.err
#BSUB -env "all"
#BSUB -B
#BSUB -N

#(!) Important note: Assumes that you run the script from the DeepIsoQ root directory. 

# Create output dir
mkdir -p Misc/results

echo "=== Job Started ==="

# Activate environment
source ../.venv/bin/activate

# Run the script
python3 Misc/plot_raw_data.py

echo "=== Job Finished ==="

#Run the job using (cd to DeepIsoQ): 
# bsub < Misc/plot_raw_data_job.sh
