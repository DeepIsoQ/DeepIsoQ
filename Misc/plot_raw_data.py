#02/12-2025

#The purpose of this script is to make plots of the initial gene and transcript data distribution. 

#Be sure to have copied the ddata over to the BLACKHOLE directory. 
#This script also assumes that you have run the data preprocessing script to create data.pt


import os
import torch
import numpy as np
import matplotlib
# Force matplotlib to use 'Agg' backend (headless/no monitor)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ------------------------------
# Config
# ------------------------------
OUTPUT_DIR = "Misc/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Data path (Portable)
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")

if DATA_PT is None:
    # Fall back to $BLACKHOLE/$USER/data.pt
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export DATA_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
# Load on CPU to save GPU resources
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

###########################################################################

def analyze_and_plot(tensor_data, name, is_already_log=False):
    """
    Computes stats and plots histograms for a given tensor.
    """
    print("\n" + "="*60)
    print(f"{name} DATA")
    print("="*60)
    
    # 1. Basic Shape
    N, F = tensor_data.shape
    print(f"Shape: {tensor_data.shape}")
    print(f"  Samples: {N}")
    print(f"  Features: {F}")

    # 2. Statistics
    # We use .view(-1) to flatten without copying memory (if contiguous)
    # converting to numpy makes stat calculations easier/standard
    flat_data = tensor_data.view(-1).numpy()
    
    print("[INFO] Calculating statistics...")
    zero_count = (flat_data == 0).sum()
    zero_pct = (zero_count / flat_data.size) * 100
    
    print(f"Sparsity: {zero_pct:.2f}% zeros")
    print(f"Min: {flat_data.min():.4f}")
    print(f"Max: {flat_data.max():.4f}")
    print(f"Mean: {flat_data.mean():.4f}")
    print(f"Median: {np.median(flat_data):.4f}")

    # 3. Plotting
    print(f"[INFO] Generating plots for {name}...")
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: The data as it is stored
    plt.subplot(1, 2, 1)
    # Use bins='auto' or 100. Log scale Y helps visualize the 'long tail'
    plt.hist(flat_data, bins=100, color='skyblue', edgecolor='black')
    label = "Log1p Values" if is_already_log else "Raw Values"
    plt.xlabel(label)
    plt.ylabel('Frequency (Log Scale)')
    plt.yscale('log')
    plt.title(f'{name} Distribution (As Stored)')

    # Subplot 2: Log Transformation
    plt.subplot(1, 2, 2)
    if is_already_log:
        # If already log, plot without log-scale Y axis to see "shape" better
        plt.hist(flat_data, bins=100, color='salmon', edgecolor='black')
        plt.xlabel('Log1p Values')
        plt.ylabel('Frequency (Linear Scale)')
        plt.title(f'{name} Shape (Linear Y-Axis)')
    else:
        # If raw, show what it looks like logged
        # We add 1e-9 to avoid log(0) errors if not using log1p
        log_data = np.log1p(flat_data)
        plt.hist(log_data, bins=100, color='salmon', edgecolor='black')
        plt.xlabel('Log1p(Values)')
        plt.ylabel('Frequency')
        plt.title(f'{name} Distribution (Log1p Transformed)')

    plt.tight_layout()
    
    # Save
    safe_name = name.lower().replace(" ", "_")
    save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_dist.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Plot saved to {save_path}")

# ------------------------------
# Execution
# ------------------------------

# 1. Process Genes (Xg_log1p)
# Note: These are ALREADY log1p transformed
if "Xg_log1p" in data:
    X = data["Xg_log1p"].float() # Ensure float
    analyze_and_plot(X, "GENES (Xg_log1p)", is_already_log=True)
else:
    print("[WARN] Key 'Xg_log1p' not found in data.pt")

# 2. Process Transcripts (Y_tx)
# Note: These are usually Raw counts in your previous scripts
if "Y_tx" in data:
    Y = data["Y_tx"].float() # Ensure float
    analyze_and_plot(Y, "TRANSCRIPTS (Y_tx)", is_already_log=False)
else:
    print("[WARN] Key 'Y_tx' not found in data.pt")

print("\n[INFO] Analysis complete.")