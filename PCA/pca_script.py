import os
import json
import time
import csv
import torch
import numpy as np
import matplotlib
# [CRITICAL] Force matplotlib to use non-interactive backend for HPC
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

PCA_DIM         = 1000  # Number of PCA components
BATCH_SIZE      = 128
N_EPOCHS        = 30
LR              = 1e-3

# ------------------------------
# Dynamic Output Paths
# ------------------------------
# Try to get the LSF Job ID. If running locally, use a timestamp.
JOB_ID = os.environ.get("LSB_JOBID")
if JOB_ID is None:
    JOB_ID = f"local_{time.strftime('%Y%m%d_%H%M%S')}"

print(f"[INFO] Detected Job ID: {JOB_ID}")

OUT_DIR         = "PCA/results"
# Append Job ID to filenames so they don't overwrite each other
FIG_PATH        = os.path.join(OUT_DIR, f"pca_training_curve_{JOB_ID}.png")
JSON_PATH       = os.path.join(OUT_DIR, f"pca_metrics_{JOB_ID}.json")
CSV_PATH        = os.path.join(OUT_DIR, f"pca_training_log_{JOB_ID}.csv")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Seed | Ensuring reproducibility
# ------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ------------------------------
# Data Loading
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

# ------------------------------
# Data Processing
# ------------------------------
print("[INFO] Splitting, Scaling, and running PCA...")

# 1. Split
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True
)
relative_val_size = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=relative_val_size, random_state=SEED, shuffle=True
)

# 2. Standardize
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val   = scaler_x.transform(X_val)
X_test  = scaler_x.transform(X_test)

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# 3. PCA
pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train)
X_val_pca   = pca.transform(X_val)
X_test_pca  = pca.transform(X_test)

# 4. To Tensor
X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_pca,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_pca,  dtype=torch.float32)

Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32)

# 5. DataLoaders
train_dataset = TensorDataset(X_train_t, Y_train_t)
val_dataset   = TensorDataset(X_val_t,   Y_val_t)
test_dataset  = TensorDataset(X_test_t,  Y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Training batches: {len(train_loader)} | Val batches: {len(val_loader)}")

# ------------------------------
# Model
# ------------------------------
class IsoformPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Training Setup
# ------------------------------
input_dim = X_train_t.shape[1]
output_dim = Y_train_t.shape[1]

model = IsoformPredictor(input_dim, output_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss() 

# ------------------------------
# Training Loop
# ------------------------------
print(f"[INFO] Starting training on {DEVICE}...")
train_losses = []
val_losses = []
start_time = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    total_train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1:02d}/{N_EPOCHS}] | Train Loss (Scaled): {avg_train_loss:.4f} | Val Loss (Scaled): {avg_val_loss:.4f}")

total_time = time.time() - start_time

# ------------------------------
# Final Test Evaluation (UNSCALED)
# ------------------------------

#Note: In order to make the model comparable to the non-PCA script, we need to inverse transform
#the predictions back to the original log1p scale before calculating the final MSE! 

print("\n[INFO] Running Final Test Evaluation (Unscaling back to log1p)...")
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        # Get Scaled Predictions
        preds_scaled = model(xb)
        
        # Move to CPU and store
        all_preds.append(preds_scaled.cpu().numpy())
        all_targets.append(yb.numpy()) # yb was already on CPU in DataLoader, or move .cpu() if needed

# 1. Concatenate all batches
P_scaled = np.vstack(all_preds)
T_scaled = np.vstack(all_targets)

# 2. Inverse Transform (Back to original log1p scale)
# This makes it comparable to your other non-scaled script
P_log1p = scaler_y.inverse_transform(P_scaled)
T_log1p = scaler_y.inverse_transform(T_scaled)

# 3. Calculate MSE on the unscaled data
final_test_mse_unscaled = ((P_log1p - T_log1p)**2).mean()

print(f"[FINAL] Test MSE (Scaled):   {((P_scaled - T_scaled)**2).mean():.4f}")
print(f"[FINAL] Test MSE (Unscaled) - Comparable version: {final_test_mse_unscaled:.4f}")

# ------------------------------
# Save Results & Plots
# ------------------------------

# 1. Save Training Curve Plot
try:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss (Scaled)', marker='o')
    plt.plot(val_losses, label='Validation Loss (Scaled)', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Scaled)')
    #plt.title(f'PCA FFNN Training (PCA={PCA_DIM})')
    plt.title(f'PCA FFNN (PCA={PCA_DIM}) | Job: {JOB_ID}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved to: {FIG_PATH}")
except Exception as e:
    print(f"[WARN] Failed to save plot: {e}")

# 2. Save Metrics to JSON
summary = {
    "meta": {
        "job_id": JOB_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    },
    "config": {
        "pca_dim": PCA_DIM,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": N_EPOCHS,
        "seed": SEED
    },
    "results": {
        "final_train_mse_scaled": train_losses[-1],
        "final_val_mse_scaled": val_losses[-1],
        "test_mse_scaled": float(((P_scaled - T_scaled)**2).mean()),
        "test_mse_unscaled": float(final_test_mse_unscaled), # This is the key metric for model comparison
        "total_time_sec": round(total_time, 2)
    }
}
with open(JSON_PATH, "w") as f:
    json.dump(summary, f, indent=4)
print(f"[INFO] Metrics saved to: {JSON_PATH}")

# 3. Save Logs to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_mse_scaled", "val_mse_scaled"])
    for i in range(N_EPOCHS):
        writer.writerow([i+1, train_losses[i], val_losses[i]])

print("[INFO] All done.")
