#!/bin/bash
#BSUB -J pca_ffnn_gpu
#BSUB -q gpuv100                                # Primary queue. If slow/busy, change to: gpul40s (https://www.hpc.dtu.dk/?page_id=2759)
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "rusage[mem=64GB] span[hosts=1]"  # 64GB System RAM for running the script. 
#BSUB -cwd .
#BSUB -o PCA/logs/pca_ffnn_%J.out
#BSUB -env "all"

# (optional) email
#BSUB -u s215065@dtu.dk     # <-- your email
#BSUB -B
#BSUB -N

# Safety: check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] GPU not found. Exiting."
    exit 1
fi

#Create output dirs, if they don't exist already: 
mkdir -p PCA/logs
mkdir -p PCA/results

echo "[INFO] GPU detected:"
nvidia-smi

module load cuda/12.8.1         # <-- your cuda version  (hint: you can also run 'module avail cuda'). Make sure it matches your PyTorch verison. 
source ../.venv/bin/activate          # <-- your environment path

echo "=== Starting training ==="
python3 PCA/pca_script.py

#Submit it using (cd to DeepIsoQ):
# bsub < PCA/pca_ffnn_job.sh
