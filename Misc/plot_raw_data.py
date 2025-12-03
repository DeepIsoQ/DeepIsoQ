#02/12-2025

#The purpose of this script is to make plots of the initial gene and transcript data distribution. 

#Be sure to have copied the ddata over to the BLACKHOLE directory. 
#This script also assumes that you have run the data preprocessing script to create data.pt


import os
import time
import torch
import numpy as np
import matplotlib
# Force matplotlib to use 'Agg' backend (headless/no monitor)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ------------------------------
# Fingerprinting (Job ID)
# ------------------------------
JOB_ID = os.environ.get("LSB_JOBID")
if JOB_ID is None:
    JOB_ID = f"local_{time.strftime('%Y%m%d_%H%M%S')}"

print(f"[INFO] Detected Job ID: {JOB_ID}")

# ------------------------------
# Config
# ------------------------------
OUTPUT_DIR = "Misc/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Data path
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")

if DATA_PT is None:
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        # Fallback default
        DATA_PT = "data.pt"
    else:
        DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
# Load on CPU to save GPU resources
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

# ------------------------------
# Data Preparation (Standardizing to Log1p)
# ------------------------------
print("[INFO] Preparing data...")

# 1. Genes (Already Log1p)
if "Xg_log1p" in data:
    genes_flat = data["Xg_log1p"].float().view(-1).numpy()
    print(f"Genes loaded. Shape: {data['Xg_log1p'].shape}")
else:
    raise ValueError("Xg_log1p missing from data.pt")

# 2. Transcripts (Raw -> Convert to Log1p for fair comparison)
if "Y_tx" in data:
    # We apply log1p here so we compare the same scale for both genes and transcripts in the plots
    transcripts_raw = data["Y_tx"].float()
    transcripts_flat = torch.log1p(transcripts_raw).view(-1).numpy()
    print(f"Transcripts loaded and log-transformed. Shape: {data['Y_tx'].shape}")
else:
    raise ValueError("Y_tx missing from data.pt")

# ------------------------------
# Statistics Calculation
# ------------------------------
def get_stats(flat_data):
    zero_count = (flat_data == 0).sum()
    return {
        "sparsity": (zero_count / flat_data.size) * 100,
        "max": flat_data.max(),
        "mean": flat_data.mean(),
        "median": np.median(flat_data)
    }

gene_stats = get_stats(genes_flat)
tx_stats = get_stats(transcripts_flat)

print(f"\nGENES (Log1p): {gene_stats}")
print(f"TRANSCRIPTS (Log1p): {tx_stats}")

# ------------------------------
# Combined Plotting (2x2 Grid)
# ------------------------------
print("[INFO] Generating Comparison Plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Add Global Title with Job ID
fig.suptitle(f"Gene vs Transcript Distribution (Log1p) | Job: {JOB_ID}", fontsize=16)

# Colors
GENE_COL = 'skyblue'
TX_COL = 'salmon'
BINS = 100

# --- ROW 1: Linear Y-Axis (The Shape) ---

# Top-Left: Genes
axes[0, 0].hist(genes_flat, bins=BINS, color=GENE_COL, edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f"Genes (Input)\nSparsity: {gene_stats['sparsity']:.1f}%")
axes[0, 0].set_ylabel("Frequency (Linear)")
axes[0, 0].set_xlabel("Expression (Log1p)")

# Top-Right: Transcripts
axes[0, 1].hist(transcripts_flat, bins=BINS, color=TX_COL, edgecolor='black', alpha=0.7)
axes[0, 1].set_title(f"Transcripts (Target)\nSparsity: {tx_stats['sparsity']:.1f}%")
axes[0, 1].set_ylabel("Frequency (Linear)")
axes[0, 1].set_xlabel("Expression (Log1p)")

# --- ROW 2: Log Y-Axis (The Tails) ---

# Bottom-Left: Genes
axes[1, 0].hist(genes_flat, bins=BINS, color=GENE_COL, edgecolor='black', alpha=0.7)
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel("Frequency (Log Scale)")
axes[1, 0].set_xlabel("Expression (Log1p)")
axes[1, 0].set_title("Gene Distribution (Log Y-Axis)")
axes[1, 0].grid(True, which="both", ls="-", alpha=0.2)

# Bottom-Right: Transcripts
axes[1, 1].hist(transcripts_flat, bins=BINS, color=TX_COL, edgecolor='black', alpha=0.7)
axes[1, 1].set_yscale('log')
axes[1, 1].set_ylabel("Frequency (Log Scale)")
axes[1, 1].set_xlabel("Expression (Log1p)")
axes[1, 1].set_title("Transcript Distribution (Log Y-Axis)")
axes[1, 1].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

# Save with Job ID
filename = f"data_distribution_comparison_{JOB_ID}.png"
save_path = os.path.join(OUTPUT_DIR, filename)
plt.savefig(save_path, dpi=300) 
plt.close()

print(f"[SUCCESS] Comparison plot saved to {save_path}")