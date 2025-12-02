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

# Output Paths
OUT_DIR         = "PCA/results"
FIG_PATH        = os.path.join(OUT_DIR, "pca_training_curve.png")
JSON_PATH       = os.path.join(OUT_DIR, "pca_metrics.json")
CSV_PATH        = os.path.join(OUT_DIR, "pca_training_log.csv")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Reproducibility
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
        # Fallback for local testing if env vars missing
        DATA_PT = "data.pt" 
    else:
        DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

# Load data
X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

#######################################################################

# ------------------------------
# Data Processing (Split -> Scale -> PCA)
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

# 2. Standardize (Fit on Train ONLY)
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val   = scaler_x.transform(X_val)
X_test  = scaler_x.transform(X_test)

scaler_y = StandardScaler()
Y_train = scaler_y.fit_transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# 3. PCA (Fit on Train ONLY)
pca = PCA(n_components=PCA_DIM)
X_train_pca = pca.fit_transform(X_train)
X_val_pca   = pca.transform(X_val)
X_test_pca  = pca.transform(X_test)

print(f"[INFO] Input shape after PCA: {X_train_pca.shape}")

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

###################################################################################

# ------------------------------
# Model Definition
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
    # --- Training ---
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

    print(f"Epoch [{epoch+1:02d}/{N_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

total_time = time.time() - start_time

# ------------------------------
# Final Test Evaluation
# ------------------------------
print("\n[INFO] Running Final Test Evaluation...")
model.eval()
total_test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_test_loss += loss.item()

final_test_mse = total_test_loss / len(test_loader)
print(f"[FINAL] Test MSE: {final_test_mse:.4f}")

# ------------------------------
# Save Results & Plots
# ------------------------------

# 1. Save Training Curve Plot
try:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'PCA FFNN Training (PCA={PCA_DIM})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()
    print(f"[INFO] Plot saved to: {FIG_PATH}")
except Exception as e:
    print(f"[WARN] Failed to save plot: {e}")

# 2. Save Metrics to JSON
summary = {
    "config": {
        "pca_dim": PCA_DIM,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": N_EPOCHS,
        "seed": SEED
    },
    "results": {
        "final_train_mse": train_losses[-1],
        "final_val_mse": val_losses[-1],
        "test_mse": final_test_mse,
        "total_time_sec": round(total_time, 2)
    }
}
with open(JSON_PATH, "w") as f:
    json.dump(summary, f, indent=4)
print(f"[INFO] Metrics saved to: {JSON_PATH}")

# 3. Save Logs to CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_mse", "val_mse"])
    for i in range(N_EPOCHS):
        writer.writerow([i+1, train_losses[i], val_losses[i]])

print("[INFO] All done.")