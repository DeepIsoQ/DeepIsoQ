#!/usr/bin/env python3

#04/12-2025

#This was the last PCA-based model script to be made. Its sole purpose is to re-run the best hyperparameter
#configuration found during previous experiments (multiple_pca_ffnn).
#It also has the purpose of generating some additional final metrics. 

"""
PCA FFNN - Best Model Training (GPU + AMP)
Re-runs the specific best configuration (L1_512_t53) to generate final metrics.
"""

import os
import json
import time
import random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for HPC
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# 1. Configuration (matches L1_512_t53, the best model)
# ------------------------------
BEST_HP = {
    "pca_dim": 1000,
    "hidden": [512],      # One layer of 512
    "act": "tanh",        # tanh activation function
    "lr": 0.003,          # Learning rate 
    "batch_size": 128,    
    "dropout": 0.0,       # No dropout
    "batchnorm": True,    
    "epochs": 50          
}

SEED      = 42
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP       = (DEVICE == "cuda")

# Output Paths
JOB_ID = os.environ.get("LSB_JOBID", "local")
OUT_DIR = "PCA/results_final"
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_PATH  = os.path.join(OUT_DIR, "training_curve_standard.png")
JSON_PATH  = os.path.join(OUT_DIR, "final_metrics.json")

# ------------------------------
# 2. Helper Functions
# ------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_activation(name: str):
    if name == "relu": return nn.ReLU()
    if name == "gelu": return nn.GELU()
    if name == "tanh": return nn.Tanh()
    if "leaky" in name: return nn.LeakyReLU(0.01)
    return nn.ReLU()

def pearson_corr_gpu(y_pred, y_true):
    vx = y_pred - torch.mean(y_pred, dim=0)
    vy = y_true - torch.mean(y_true, dim=0)
    cost = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + 1e-8)
    return torch.nanmean(cost).item()

# ------------------------------
# 3. Data Loading & Processing
# ------------------------------
set_seed(SEED)

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

print("[INFO] Splitting, Scaling, PCA...")

# Split
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
rel_val = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=rel_val, random_state=SEED, shuffle=True)

# Scale
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_val   = scaler_x.transform(X_val)
X_test  = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(Y_train)
Y_train = scaler_y.transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# PCA
pca = PCA(n_components=BEST_HP["pca_dim"]).fit(X_train)
X_train = pca.transform(X_train)
X_val   = pca.transform(X_val)
X_test  = pca.transform(X_test)

# Tensor Setup
Xt_train = torch.tensor(X_train, dtype=torch.float32)
Yt_train = torch.tensor(Y_train, dtype=torch.float32)
Xt_val   = torch.tensor(X_val,   dtype=torch.float32)
Yt_val   = torch.tensor(Y_val,   dtype=torch.float32)
Xt_test  = torch.tensor(X_test,  dtype=torch.float32)
Yt_test  = torch.tensor(Y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(Xt_train, Yt_train), batch_size=BEST_HP["batch_size"], shuffle=True, drop_last=True)
val_loader   = DataLoader(TensorDataset(Xt_val, Yt_val),     batch_size=BEST_HP["batch_size"], shuffle=False)
test_loader  = DataLoader(TensorDataset(Xt_test, Yt_test),   batch_size=BEST_HP["batch_size"], shuffle=False)

# ------------------------------
# 4. Model Definition
# ------------------------------
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

# ------------------------------
# 5. Training Loop
# ------------------------------
print(f"[INFO] Starting training for {BEST_HP['epochs']} epochs...")

model = FFNN(BEST_HP["pca_dim"], Y.shape[1], BEST_HP).to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=BEST_HP["lr"], weight_decay=1e-4)
criterion = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=AMP)

history = {"train_mse": [], "val_mse": []}

t0 = time.time()

for epoch in range(1, BEST_HP["epochs"] + 1):
    # --- TRAIN ---
    model.train()
    loss_sum, counts = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            preds = model(xb)
            loss = criterion(preds, yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss_sum += loss.item() * yb.size(0)
        counts += yb.size(0)
    train_mse = loss_sum / counts
    history["train_mse"].append(train_mse)

    # --- VALIDATE ---
    model.eval()
    vloss_sum, vcounts = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p = model(xb)
            vloss_sum += criterion(p, yb).item() * yb.size(0)
            vcounts += yb.size(0)
    val_mse = vloss_sum / vcounts
    history["val_mse"].append(val_mse)

    print(f"Ep {epoch:02d} | Tr: {train_mse:.4f} | Val: {val_mse:.4f}")

# ------------------------------
# 6. Final Evaluation (Scaled MSE, Unscaled MSE, Pearson)
# ------------------------------
print("\n[INFO] Calculating Final Metrics for Table...")

model.eval()

def get_comprehensive_metrics(loader):
    preds_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            p = model(xb)
            preds_list.append(p.cpu().numpy())
            true_list.append(yb.numpy()) 
    
    # Stack
    P_scaled = np.vstack(preds_list)
    T_scaled = np.vstack(true_list)
    
    # 1. Scaled MSE
    mse_scaled = ((P_scaled - T_scaled)**2).mean()

    # 2. Unscaled MSE (Inverse Transform)
    P_log1p = scaler_y.inverse_transform(P_scaled)
    T_log1p = scaler_y.inverse_transform(T_scaled)
    mse_unscaled = ((P_log1p - T_log1p)**2).mean()
    
    # 3. Pearson (on Unscaled data)
    P_t = torch.tensor(P_log1p, device=DEVICE)
    T_t = torch.tensor(T_log1p, device=DEVICE)
    rho = pearson_corr_gpu(P_t, T_t)
    
    return float(mse_scaled), float(mse_unscaled), float(rho)

val_sc, val_un, val_rho = get_comprehensive_metrics(val_loader)
test_sc, test_un, test_rho = get_comprehensive_metrics(test_loader)

print("="*60)
print("FINAL RESULTS FOR TABLE")
print("="*60)
print(f"VALIDATION:")
print(f"  MSE (Unscaled): {val_un:.4f}")
print(f"  Pearson (rho):  {val_rho:.4f}")
print("-" * 30)
print(f"TEST:")
print(f"  MSE (Unscaled): {test_un:.4f}")
print(f"  Pearson (rho):  {test_rho:.4f}")
print("="*60)


# ------------------------------
# 7. Plotting
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history["train_mse"])+1), history["train_mse"], label="Train MSE (Scaled)", marker='.')
plt.plot(range(1, len(history["val_mse"])+1), history["val_mse"], label="Val MSE (Scaled)", marker='.')
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Scaled)")
plt.title(f"Training Curve (PCA={BEST_HP['pca_dim']})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOT_PATH, dpi=300)
plt.close()

print(f"[SUCCESS] Done. Plot: {PLOT_PATH}")


# Save metrics
with open(JSON_PATH, "w") as f:
    json.dump({
        "hp": BEST_HP,
        "validation": {"mse_scaled": val_sc, "mse_unscaled": val_un, "rho": val_rho},
        "test":       {"mse_scaled": test_sc, "mse_unscaled": test_un, "rho": test_rho}
    }, f, indent=4)

print("[SUCCESS] Final metrics saved to:", JSON_PATH)
