#!/bin/bash
#BSUB -J pca_dim_search
#BSUB -q gpul40s            #Alternatively, use gpuv100.
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=64GB] span[hosts=1]"
#BSUB -cwd .
#BSUB -o PCA/logs/pca_sweep_%J.out
#BSUB -env "all"
#BSUB -u s215065@dtu.dk
#BSUB -B
#BSUB -N


# Safety checks
mkdir -p PCA/logs
mkdir -p PCA/results

if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] GPU not found."
    exit 1
fi

module load cuda/12.8.1
source ../.venv/bin/activate

echo "=== Starting PCA Dimension Sweep ==="
python3 PCA/pca_dim_search.py

#Run the job script (assumes that you are in DeepIsoQ directory): 
# bsub < PCA/pca_dim_search_job.sh
