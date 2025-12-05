#!/usr/bin/env python3

#03/12-2025

#The purpose of this script is similar to multiple_pca_script.py, 
#but where we test out 5 different values for the number of principal components to retain in PCA.
#I.e. how many dimensions we reduce to. 


"""
PCA Dimension Sweep (GPU + AMP)
Tests how model performance changes with different input dimensions (PCA components).
Keeps model hyperparameters FIXED.

Output: Metrics unscaled back to log1p space.
"""

import os
import json
import time
import csv
import random
import pathlib
import math

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib
# Safe backend for HPC (No screen)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# Config
# ------------------------------
SEED      = 42
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP       = (DEVICE == "cuda")

# ---- EXPERIMENT SETUP ----
#These are the PCA dimensions (number of PCA components) which will be tested:
PCA_DIMS_TO_TEST = [100, 500, 1000, 2000, 3000, 5000, 10000]

#The script will try to find the PCA components which give the best performance. 


#Fixed Hyperparameters - the only thing which will change is the PCA input dimensions.
#These fixed hyperparameters were chosen based on prior experimentation and a small mix of arbitrary choices. 

FIXED_HP = {
    "hidden": [1024, 512],
    "act": "gelu",
    "lr": 1e-3,
    "batch_size": 128,
    "dropout": 0.1,
    "batchnorm": True,
    "epochs": 30
}

GRAD_CLIP  = 1.0
EVAL_EVERY = 5

# ------------------------------
# Output Paths
# ------------------------------
JOB_ID = os.environ.get("LSB_JOBID")
if JOB_ID is None:
    JOB_ID = f"local_{time.strftime('%Y%m%d_%H%M%S')}"

print(f"[INFO] Detected Job ID: {JOB_ID}")

BASE_DIR = "PCA/results"
os.makedirs(BASE_DIR, exist_ok=True)
TRIAL_FIG_DIR = os.path.join(BASE_DIR, f"figs_pca_sweep_{JOB_ID}")

RESULTS_CSV  = os.path.join(BASE_DIR, f"pca_sweep_results_{JOB_ID}.csv")
SUMMARY_JSON = os.path.join(BASE_DIR, f"pca_sweep_summary_{JOB_ID}.json")
SWEEP_PLOT   = os.path.join(BASE_DIR, f"pca_sweep_plot_{JOB_ID}.png")

# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ------------------------------
# Data Loading & PCA Processing
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")
if DATA_PT is None:
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        DATA_PT = "data.pt"
    else:
        DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

print("[INFO] Processing Data...")

# 1. Split
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
rel_val = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=rel_val, random_state=SEED, shuffle=True)

# 2. Scale
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_val   = scaler_x.transform(X_val)
X_test  = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(Y_train)
Y_train = scaler_y.transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# 3. PCA (Using the highest PCA dimension to fit once)
max_dim = max(PCA_DIMS_TO_TEST)
print(f"[INFO] Fitting PCA with n_components={max_dim} (Max required)...")
pca = PCA(n_components=max_dim).fit(X_train) #We only wanna calculate the PCA once. Therefore we are using the max dimensions. 

X_train_pca = pca.transform(X_train)
X_val_pca   = pca.transform(X_val)
X_test_pca  = pca.transform(X_test)

# 4. To Tensor (again, remember, we are using the max dimensions here, as there is no point in recalculating PCA multiple times)
Xt_train_full = torch.tensor(X_train_pca, dtype=torch.float32)
Xt_val_full   = torch.tensor(X_val_pca,   dtype=torch.float32)
Xt_test_full  = torch.tensor(X_test_pca,  dtype=torch.float32)

Yt_train = torch.tensor(Y_train, dtype=torch.float32)
Yt_val   = torch.tensor(Y_val,   dtype=torch.float32)
Yt_test  = torch.tensor(Y_test,  dtype=torch.float32)

# ------------------------------
# Model Definition
# ------------------------------

#Defining activation function and FFNN class - the model will be a bit further defined in the train_and_evaluate function:

def get_activation(name: str):
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if "leaky" in name: return nn.LeakyReLU(0.01)
    return nn.ReLU()

class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hp):
        super().__init__()
        layers, prev = [], in_dim
        for h in hp["hidden"]:
            layers.append(nn.Linear(prev, h))
            if hp["batchnorm"]: layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(hp["act"]))
            if hp["dropout"] > 0: layers.append(nn.Dropout(hp["dropout"]))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

##########################################

# ------------------------------
# Training Function
# ------------------------------
def train_and_evaluate(current_pca_dim):
    print(f"\n>>> Starting Run: PCA_DIM = {current_pca_dim}")
    
    # Slice Tensors for current dimension
    xt_tr = Xt_train_full[:, :current_pca_dim]
    xt_va = Xt_val_full[:,   :current_pca_dim]
    xt_te = Xt_test_full[:,  :current_pca_dim]
    
    # Loaders
    tr_loader = DataLoader(TensorDataset(xt_tr, Yt_train), batch_size=FIXED_HP["batch_size"], shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(xt_va, Yt_val),   batch_size=FIXED_HP["batch_size"], shuffle=False)
    te_loader = DataLoader(TensorDataset(xt_te, Yt_test),  batch_size=FIXED_HP["batch_size"], shuffle=False)

    #Model - defines the model, optimizer, scheduler, criterion, scaler. 
    model = FFNN(current_pca_dim, Y.shape[1], FIXED_HP).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=FIXED_HP["lr"], weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    #Loop for training: 
    train_curve, val_curve = [], []
    t0 = time.time()
    
    for epoch in range(1, FIXED_HP["epochs"] + 1):
        model.train()
        loss_sum, counts = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                preds = model(xb)
                loss = criterion(preds, yb)
            scaler.scale(loss).backward()
            if GRAD_CLIP:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * yb.size(0)
            counts += yb.size(0)
        train_curve.append(loss_sum/counts)

        #Validation
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == FIXED_HP["epochs"]:
            model.eval()
            vloss_sum, vcounts = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    p = model(xb)
                    vloss_sum += criterion(p, yb).item() * yb.size(0)
                    vcounts += yb.size(0)
            val_loss = vloss_sum/vcounts
            val_curve.append(val_loss)
            sch.step(val_loss)
            print(f"[PCA {current_pca_dim}] ep {epoch:02d} | val_scaled {val_loss:.4f}")

    #Final Test (on unscaled data)
    model.eval()
    preds_list, targs_list = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(DEVICE)
            p = model(xb)
            preds_list.append(p.cpu().numpy())
            targs_list.append(yb.numpy())
    
    P_scaled = np.vstack(preds_list)
    T_scaled = np.vstack(targs_list)
    
    #Inverse Transform to log1p space (Back to original scale)
    P_log1p = scaler_y.inverse_transform(P_scaled)
    T_log1p = scaler_y.inverse_transform(T_scaled)
    
    final_mse = ((P_log1p - T_log1p)**2).mean()
    
    #Save training curve for this specific PCA dimension: 
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(train_curve, label="Train")
    plt.plot(np.linspace(1, FIXED_HP["epochs"], len(val_curve)), val_curve, "o-", label="Val")
    plt.title(f"PCA Dim: {current_pca_dim}")
    plt.legend()
    plt.savefig(os.path.join(TRIAL_FIG_DIR, f"curve_pca_{current_pca_dim}.png"))
    plt.close()

    #Explicitly cast to standard python float() for JSON safety. Otherwise numpy float types will cause an error!!
    return {
        "pca_dim": int(current_pca_dim), #Cast to int
        "test_mse_unscaled": float(final_mse), #Cast to float
        "test_mse_scaled": float(((P_scaled - T_scaled)**2).mean()), #Cast to float
        "time_sec": round(time.time() - t0, 1)
    }

#############################################

# ------------------------------
# Main Sweep Loop - test out the PCA dimensions that were inputted in the config section
# ------------------------------
results = []

# CSV Init
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pca_dim", "test_mse_unscaled", "test_mse_scaled", "time_sec"])
    w.writeheader()

print("\n=== STARTING PCA SWEEP ===")
for dim in PCA_DIMS_TO_TEST:
    res = train_and_evaluate(dim)
    results.append(res)
    
    # Update CSV
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=res.keys())
        w.writerow(res)
    
    print(f"[RESULT] Dim {dim} -> Test MSE (Unscaled): {res['test_mse_unscaled']:.4f}")
 
# ------------------------------
# Summary Plot
# ------------------------------
dims = [r["pca_dim"] for r in results]
mses = [r["test_mse_unscaled"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(dims, mses, marker='o', linestyle='-', linewidth=2, color='b')
plt.xlabel("Number of PCA Components")
plt.ylabel("Test MSE (Unscaled)")
plt.title("Model Performance vs. PCA Dimension")
plt.grid(True, alpha=0.3)
plt.savefig(SWEEP_PLOT, dpi=150)
plt.close()

# Save JSON
with open(SUMMARY_JSON, "w") as f:
    json.dump({"fixed_hp": FIXED_HP, "results": results}, f, indent=4)

print(f"\n[INFO] Sweep Complete. Summary plot saved to {SWEEP_PLOT}")
