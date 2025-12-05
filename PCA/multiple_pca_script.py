#!/usr/bin/env python3

#03/12-2025


"""
PCA FFNN Random Search (GPU + AMP) — 2-stage
Adapted for PCA Data (1000 Dim) vs Raw Data comparison.

Stage 1: Random search (100 trials)
Stage 2: Retrain top-5 configs
Output: Metrics unscaled back to log1p space for direct comparison with raw models.
"""

import os, json, math, random, csv, time, pathlib, ast
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
# Global config
# ------------------------------
SEED      = 42
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP       = (DEVICE == "cuda")

# PCA Specific
PCA_DIM   = 1000

# ---- STAGE CONTROL ----
# STAGE = 1 | random search with pruning
# STAGE = 2 | retrain top-K configs from Stage 1 (no global pruning)

STAGE        = 1   #<--------------------- (!) UPDATE THIS VALUE TO 2, FOR STAGE 2 RUN (!)
# 1 = Search, 2 = Retrain Top-K
JOB_ID_STAGE1 = "INSERT_JOB_ID_FROM_STAGE1_HERE"  #<--- (!) UPDATE THIS VALUE FOR STAGE 2 RUN (!)

TOP_K_STAGE2 = 5          

GRAD_CLIP    = 1.0
EVAL_EVERY   = 5          

#Hyperparameter optimization, search ranges (same as ffnn_search_v2.py): 
BATCHES    = [128, 192, 256]
LRS        = [1e-3, 2e-3, 3e-3]
DROPOUTS   = [0.0, 0.1, 0.2]
BATCHNORMS = [False, True]
ACTS       = ["tanh", "relu", "gelu", "leakyrelu"]
DEPTH_CHOICES = [1, 2, 3]
WIDTH_CHOICES = [256, 512, 1024, 2048]

if STAGE == 1:
    N_TRIALS        = 100
    MAX_EPOCHS      = 50
    PATIENCE        = 5
    MIN_PRUNE_EPOCH = 30
    PRUNE_FACTOR    = 1.5
elif STAGE == 2:
    N_TRIALS        = TOP_K_STAGE2
    MAX_EPOCHS      = 200
    PATIENCE        = 10
    MIN_PRUNE_EPOCH = None
    PRUNE_FACTOR    = None

# ------------------------------
# Dynamic Output Paths (fingerprinting based on job id)
# ------------------------------
# Use Job ID to prevent overwriting results if you run multiple jobs
JOB_ID = os.environ.get("LSB_JOBID")
if JOB_ID is None:
    JOB_ID = f"local_{time.strftime('%Y%m%d_%H%M%S')}"

print(f"[INFO] Detected Job ID: {JOB_ID}")

BASE_DIR = "PCA/results"
os.makedirs(BASE_DIR, exist_ok=True)
TRIAL_FIG_DIR   = os.path.join(BASE_DIR, f"figs_trials_{JOB_ID}")

# Define paths based on Stage + Job ID
if STAGE == 1:
    RESULTS_CSV      = os.path.join(BASE_DIR, f"pca_results_stage1_{JOB_ID}.csv")
    SUMMARY_JSON     = os.path.join(BASE_DIR, f"pca_summary_stage1_{JOB_ID}.json")
    BEST_MODEL_PT    = os.path.join(BASE_DIR, f"pca_best_model_stage1_{JOB_ID}.pt")
    SUMMARY_FIG_BAR  = os.path.join(BASE_DIR, f"pca_bar_stage1_{JOB_ID}.png")
    SUMMARY_FIG_TOP5 = os.path.join(BASE_DIR, f"pca_top5_stage1_{JOB_ID}.png")
elif STAGE == 2:
    RESULTS_CSV      = os.path.join(BASE_DIR, f"pca_results_stage2_{JOB_ID}.csv")
    SUMMARY_JSON     = os.path.join(BASE_DIR, f"pca_summary_stage2_{JOB_ID}.json")
    BEST_MODEL_PT    = os.path.join(BASE_DIR, f"pca_best_model_stage2_{JOB_ID}.pt")
    SUMMARY_FIG_BAR  = os.path.join(BASE_DIR, f"pca_bar_stage2_{JOB_ID}.png")
    SUMMARY_FIG_TOP5 = os.path.join(BASE_DIR, f"pca_top5_stage2_{JOB_ID}.png")

# IMPORTANT: To run Stage 2, you must point this to the SPECIFIC CSV file from Stage 1

# Update this filename manually AFTER Stage 1 finishes!
STAGE1_INPUT_CSV = f"PCA/results/pca_results_stage1_{JOB_ID_STAGE1}.csv" #Remember to update for stage 2, in the config section (!)

# ------------------------------
# Reproducibility and setup
# ------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

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

########################################################################

#Data processing: Split -> Scale -> PCA -> To Tensor -> DataLoaders

print("[INFO] Processing Data (Split -> Scale -> PCA)...")

# 1. Split
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
rel_val = VAL_FRAC / (1 - TEST_FRAC)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=rel_val, random_state=SEED, shuffle=True)

# 2. Scale (StandardScaler on X AND Y)
# We scale Y to help the neural network train better, but we will unscale later for comparison
scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_val   = scaler_x.transform(X_val)
X_test  = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(Y_train)
Y_train = scaler_y.transform(Y_train)
Y_val   = scaler_y.transform(Y_val)
Y_test  = scaler_y.transform(Y_test)

# 3. PCA
pca = PCA(n_components=PCA_DIM).fit(X_train)
X_train = pca.transform(X_train)
X_val   = pca.transform(X_val)
X_test  = pca.transform(X_test)

print(f"[INFO] Data ready. Input dim: {X_train.shape[1]}")

# 4. To Tensor (Keep on CPU, move batches to GPU)
Xt_train = torch.tensor(X_train, dtype=torch.float32)
Yt_train = torch.tensor(Y_train, dtype=torch.float32)
Xt_val   = torch.tensor(X_val,   dtype=torch.float32)
Yt_val   = torch.tensor(Y_val,   dtype=torch.float32)
Xt_test  = torch.tensor(X_test,  dtype=torch.float32)
Yt_test  = torch.tensor(Y_test,  dtype=torch.float32)

# 5. DataLoaders
train_ds = TensorDataset(Xt_train, Yt_train)
val_ds   = TensorDataset(Xt_val, Yt_val)
test_ds  = TensorDataset(Xt_test, Yt_test)

def get_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=(shuffle and len(ds)>bs))

#########################################################################################

#Defining the model: 

# ------------------------------
# Model
# ------------------------------
def get_activation(name: str):
    n = name.lower()
    if n == "relu": return nn.ReLU()
    if n == "gelu": return nn.GELU()
    if n in ("leakyrelu", "leaky_relu"): return nn.LeakyReLU(0.01)
    if n == "tanh": return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, act="gelu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(act))
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Hyperparameter optimization helpers
# ------------------------------
def sample_arch(rnd):
    depth = rnd.choice(DEPTH_CHOICES) #Randomly choose depth of model (amount of hidden layers). 
    hidden = [rnd.choice(WIDTH_CHOICES) for _ in range(depth)] #Amount of neurons in each hidden layer, randomly chosen. 
    return hidden

def sample_hp(t):
    rnd = random.Random(SEED + 1000 + t)
    hidden = sample_arch(rnd)
    name = f"L{len(hidden)}_" + "_".join(map(str, hidden))
    return {
        "name": f"{name}_t{t}",
        "hidden": hidden,
        "act": rnd.choice(ACTS),
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
        "batch_size": rnd.choice(BATCHES),
        "epochs": MAX_EPOCHS,
    }

####################################################################

#Training the model: 

# ------------------------------
# Train Loop
# ------------------------------
criterion = nn.MSELoss()

def train_once(hp, trial_seed, global_best_val=None, stage=1):
    set_seed(trial_seed)
    
    # Loaders specific to this batch size
    tr_loader = get_loader(train_ds, hp["batch_size"], True)
    va_loader = get_loader(val_ds, hp["batch_size"], False)

    model = FFNN(PCA_DIM, Y.shape[1], hp["hidden"], hp["act"], hp["dropout"], hp["batchnorm"]).to(DEVICE)
    
    opt = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_val = math.inf
    best_state = None
    noimp = 0
    train_curve, val_curve = [], []

    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total_loss, counts = 0.0, 0
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
            total_loss += loss.item() * yb.size(0)
            counts += yb.size(0)
        
        ep_train = total_loss / counts
        train_curve.append(ep_train)

        # Validation
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            model.eval()
            val_loss, vcounts = 0.0, 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    p = model(xb)
                    val_loss += criterion(p, yb).item() * yb.size(0)
                    vcounts += yb.size(0)
            ep_val = val_loss / vcounts
            val_curve.append(ep_val)
            sch.step(ep_val)

            if ep_val < best_val:
                best_val = ep_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1

            # Pruning (Stage 1 only)
            if stage == 1 and global_best_val < math.inf and MIN_PRUNE_EPOCH and epoch >= MIN_PRUNE_EPOCH:
                if ep_val > PRUNE_FACTOR * global_best_val:
                    print(f"[{hp['name']}] Pruned at ep {epoch}.")
                    break

            print(f"[{hp['name']}] ep {epoch:03d} | tr {ep_train:.4f} | val {ep_val:.4f}")
            
            if noimp >= PATIENCE:
                print(f"[{hp['name']}] Early stopping.")
                break

    # Save Plot
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        plt.figure(figsize=(7, 4))
        plt.plot(train_curve, label="Train (Scaled)")
        xs = [e for e in range(1, len(train_curve)+1) if e==1 or e%EVAL_EVERY==0][:len(val_curve)]
        plt.plot(xs, val_curve, "o-", label="Val (Scaled)")
        plt.title(f"{hp['name']}")
        plt.legend()
        plt.savefig(os.path.join(TRIAL_FIG_DIR, f"{hp['name']}.png"), dpi=100)
        plt.close()
    except: pass

    # Return Metrics
    # NOTE: The val_mse returned here is SCALED. 
    # This is fine for picking the best model, but not for final reporting.
    rec = {
        "name": hp["name"],
        "hidden": hp["hidden"],
        "act": hp["act"],
        "dropout": hp["dropout"],
        "batchnorm": hp["batchnorm"],
        "lr": hp["lr"],
        "batch_size": hp["batch_size"],
        "epochs": len(train_curve),
        "val_mse": float(best_val),
        "time": round(time.time() - t0, 1)
    }
    return rec, best_state, best_val


####################################################################


# ------------------------------
# Setup Stage 1 vs 2
# ------------------------------
trials_to_run = []
if STAGE == 1:
    print(f"[INFO] Stage 1: Random Search ({N_TRIALS} trials)")
    for t in range(N_TRIALS):
        trials_to_run.append(sample_hp(t))
else:
    print(f"[INFO] Stage 2: Retraining Top {N_TRIALS} from CSV")
    if not os.path.exists(STAGE1_INPUT_CSV):
        raise FileNotFoundError(f"Missing Stage 1 CSV: {STAGE1_INPUT_CSV}")
    with open(STAGE1_INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = sorted(list(reader), key=lambda r: float(r["val_mse"]))[:N_TRIALS]
    
    for i, r in enumerate(rows):
        hp = {
            "name": f"{r['name']}_S2",
            "hidden": ast.literal_eval(r["hidden"]),
            "act": r["act"],
            "dropout": float(r["dropout"]),
            "batchnorm": r["batchnorm"] == "True",
            "lr": float(r["lr"]),
            "batch_size": int(r["batch_size"]),
            "epochs": MAX_EPOCHS
        }
        trials_to_run.append(hp)

####################################################################

# ------------------------------
# Main Loop
# ------------------------------
results = []
best_rec = None
best_state_global = None
GLOBAL_BEST_VAL = math.inf

# CSV Init
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name","hidden","act","dropout","batchnorm","lr","batch_size","epochs","val_mse","time"])
    w.writeheader()

for i, hp in enumerate(trials_to_run):
    print(f"\n=== Trial {i+1}/{len(trials_to_run)}: {hp['name']} ===")
    rec, state, val_score = train_once(hp, SEED+i, GLOBAL_BEST_VAL, STAGE)
    
    # Save to CSV
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rec.keys())
        w.writerow(rec)
    
    results.append(rec)

    # Track Global Best
    if STAGE == 1:
        GLOBAL_BEST_VAL = min(GLOBAL_BEST_VAL, val_score)

    # Save Best Model Checkpoint
    if best_rec is None or val_score < best_rec["val_mse"]:
        best_rec = rec
        best_state_global = state
        # Save model + scaler stats
        torch.save({
            "state_dict": state,
            "hp": hp,
            "scaler_mean": scaler_y.mean_, # Save scaler stats to unscale later
            "scaler_scale": scaler_y.scale_,
            "meta": {"job_id": JOB_ID}
        }, BEST_MODEL_PT)
        print(f"[NEW BEST] {hp['name']} (Val MSE Scaled: {val_score:.4f})")

####################################################################

# ------------------------------
# Final Evaluation (Unscaling)
# ------------------------------
print("\n[INFO] Evaluating Best Model on Test Set (UNSCALED)...")
model = FFNN(PCA_DIM, Y.shape[1], best_rec["hidden"], best_rec["act"], best_rec["dropout"], best_rec["batchnorm"]).to(DEVICE)
model.load_state_dict(best_state_global)
model.eval()

te_loader = get_loader(test_ds, best_rec["batch_size"], False)
preds_list, targs_list = [], []

with torch.no_grad():
    for xb, yb in te_loader:
        xb = xb.to(DEVICE)
        p = model(xb)
        preds_list.append(p.cpu().numpy())
        targs_list.append(yb.numpy())

P_scaled = np.vstack(preds_list)
T_scaled = np.vstack(targs_list)

# INVERSE TRANSFORM
P_log1p = scaler_y.inverse_transform(P_scaled)
T_log1p = scaler_y.inverse_transform(T_scaled)

final_mse = ((P_log1p - T_log1p)**2).mean()

# Fast GPU Pearson
yt = torch.tensor(T_log1p, device=DEVICE)
yp = torch.tensor(P_log1p, device=DEVICE)
yt = yt - yt.mean(0, keepdim=True)
yp = yp - yp.mean(0, keepdim=True)
num = (yt * yp).sum(0)
den = torch.sqrt((yt**2).sum(0)) * torch.sqrt((yp**2).sum(0)) + 1e-8
pearson = (num / den).mean().item()

print(f"\n[FINAL RESULTS] Job: {JOB_ID}")
print(f"Best Config: {best_rec['name']}")
print(f"Test MSE (Scaled):   {((P_scaled - T_scaled)**2).mean():.4f}")
print(f"Test MSE (Unscaled) - Comparable to normal FFNN Model: {final_mse:.4f}")
print(f"Test Pearson:        {pearson:.4f}")

# Save Summary
# [FIX] Cast numpy/torch floats to standard python floats for JSON serialization
summary = {
    "stage": STAGE,
    "job_id": JOB_ID,
    "best_config": best_rec,
    "metrics": {
        "test_mse_unscaled": float(final_mse), # <--- Cast to float()
        "test_pearson": float(pearson)         # <--- Cast to float()
    }
}
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=4)

print("[INFO] Done.")