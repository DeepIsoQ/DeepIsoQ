import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# Config
# =========================
# Path to your preprocessed tensors
# If you saved to BLACKHOLE/USER/data.pt, this will work out of the box:
RUNDIR   = os.path.join(os.environ.get("BLACKHOLE", "/tmp"), os.environ.get("USER", "user"))
DATA_PT  = os.environ.get("DATA_PT", os.path.join(RUNDIR, "data.pt"))

SEED       = 42
VAL_SIZE   = 0.15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_MODE  = "mixed"    # choose: "mse" or "mixed"
LAMBDA_PROP = 0.5       # weight for proportions term in the mixed loss (0.3–0.7 reasonable)
EPOCHS     = 200
PATIENCE   = 20
LR         = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN     = [1024, 512, 256]   # try also [2048,1024] or [512,512,512,512]
DROPOUT    = 0.2
BATCHNORM  = True

# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pearson_mean(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    denom = (torch.sqrt((yt**2).sum(dim=0)) * torch.sqrt((yp**2).sum(dim=0)) + 1e-8)
    r = (yt * yp).sum(dim=0) / denom
    return float(torch.nanmean(r).item())

# =========================
# Data loading
# =========================
print(f"Loading: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu")

# X: log1p(genes) already done in preprocessing
X = data["Xg_log1p"].numpy().astype("float32")      # (N, G)
# Y: raw isoforms → log1p here to match your previous pipeline
Y = np.log1p(data["Y_tx"].numpy()).astype("float32")  # (N, I)

gene_ids = data.get("gene_ids", None)
tx_ids   = data.get("tx_ids", None)

# Optional mapping (recommended for mixed loss)
g2t   = data.get("gene_to_transcripts", None)          # dict: gene_id -> list of tx_ids
tx2ix = data.get("transcript_id_to_index", None)       # dict: tx_id  -> int column index in Y

print(f"Shapes: X={X.shape}, Y={Y.shape}")

# Standardize (fit on train only later would be ideal, but we’re following your baseline)
xscaler = StandardScaler()
yscaler = StandardScaler()
X = xscaler.fit_transform(X)
Y = yscaler.fit_transform(Y)

# Split
X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=VAL_SIZE, random_state=SEED)
X_tr = torch.tensor(X_tr, device=DEVICE)
X_va = torch.tensor(X_va, device=DEVICE)
Y_tr = torch.tensor(Y_tr, device=DEVICE)
Y_va = torch.tensor(Y_va, device=DEVICE)

# =========================
# Groups for mixed loss
# =========================
groups = []
if LOSS_MODE == "mixed":
    if (gene_ids is None) or (g2t is None) or (tx2ix is None):
        raise ValueError(
            "Mixed loss requires mapping: gene_ids, gene_to_transcripts, transcript_id_to_index present in data.pt"
        )
    for gid in gene_ids:
        tx_list = g2t.get(gid, [])
        idxs = [tx2ix[tid] for tid in tx_list if tid in tx2ix]
        if len(idxs) > 0:
            groups.append(torch.tensor(idxs, device=DEVICE, dtype=torch.long))
    if len(groups) == 0:
        raise ValueError("No groups could be formed from the mapping; check gene_to_transcripts / transcript_id_to_index")

# =========================
# Model
# =========================
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, act="relu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU() if act == "relu" else nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = FFNN(
    in_dim=X.shape[1],
    out_dim=Y.shape[1],
    hidden=HIDDEN,
    act="relu",
    dropout=DROPOUT,
    batchnorm=BATCHNORM
).to(DEVICE)

# =========================
# Losses
# =========================
mse = nn.MSELoss()

def group_softmax(logits: torch.Tensor, groups):
    # logits: (N, I)
    P = torch.zeros_like(logits)
    for idxs in groups:
        P[:, idxs] = F.softmax(logits[:, idxs], dim=1)
    return P

def targets_to_proportions(Y_abs: torch.Tensor, groups, eps=1e-8):
    P = torch.zeros_like(Y_abs)
    for idxs in groups:
        denom = Y_abs[:, idxs].sum(dim=1, keepdim=True) + eps
        P[:, idxs] = Y_abs[:, idxs] / denom
    return P

def mixed_loss(logits: torch.Tensor, Y_abs: torch.Tensor, groups, lambda_prop=0.5):
    """
    Mixed objective:
      - proportion term: KL(P_true || P_pred) within each gene
      - absolute term: MSE between reconstructed absolutes and Y_abs
    """
    # Proportions (within each gene)
    P_pred = group_softmax(logits, groups)          # (N, I), softmax per gene-group
    P_true = targets_to_proportions(Y_abs, groups)  # (N, I)

    # Reconstruct absolute predictions using ground-truth gene totals
    Y_recon = torch.zeros_like(Y_abs)
    for idxs in groups:
        totals = Y_abs[:, idxs].sum(dim=1, keepdim=True)  # (N,1)
        Y_recon[:, idxs] = P_pred[:, idxs] * totals

    # Losses
    # add small constant to avoid log(0)
    loss_prop = F.kl_div((P_true + 1e-8).log(), P_pred, reduction="batchmean")
    loss_abs  = mse(Y_recon, Y_abs)
    return lambda_prop * loss_prop + (1.0 - lambda_prop) * loss_abs

# =========================
# Train
# =========================
def train(model, Xtr, Ytr, Xva, Yva, loss_mode="mse"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=max(1, PATIENCE//2))

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits_tr = model(Xtr)
        if loss_mode == "mixed":
            loss_tr = mixed_loss(logits_tr, Ytr, groups, LAMBDA_PROP)
        else:
            loss_tr = mse(logits_tr, Ytr)
        loss_tr.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_va = model(Xva)
            if loss_mode == "mixed":
                val_loss = float(mixed_loss(logits_va, Yva, groups, LAMBDA_PROP).item())
            else:
                val_loss = float(mse(logits_va, Yva).item())
            val_r = pearson_mean(Yva, logits_va)

        sched.step(val_loss)

        improved = val_loss < best_loss - 0.0
        if improved:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train {loss_mode} loss: {loss_tr.item():.5f} | "
                  f"Val {loss_mode} loss: {val_loss:.5f} | Val r: {val_r:.4f}")

        if no_improve >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_va = model(Xva)
        final_loss = float(mse(logits_va, Yva).item())      # report MSE on absolutes for comparability
        final_r    = pearson_mean(Yva, logits_va)

    return final_loss, final_r

# =========================
# Run: try both losses
# =========================
set_seed(SEED)

print(f"\n=== Training with LOSS_MODE='{LOSS_MODE}' ===")
mse_val, r_val = train(model, X_tr, Y_tr, X_va, Y_va, loss_mode=LOSS_MODE)
print(f"\n[{LOSS_MODE.upper()}] Final Val MSE: {mse_val:.6f} | Val Pearson: {r_val:.4f}")

# If you also want to compare quickly with pure MSE in the same run:
if LOSS_MODE == "mixed":
    print("\n=== Training a fresh model with LOSS_MODE='mse' (baseline) ===")
    model2 = FFNN(
        in_dim=X.shape[1], out_dim=Y.shape[1],
        hidden=HIDDEN, act="relu", dropout=DROPOUT, batchnorm=BATCHNORM
    ).to(DEVICE)
    set_seed(SEED)  # fair comparison
    mse_val2, r_val2 = train(model2, X_tr, Y_tr, X_va, Y_va, loss_mode="mse")
    print(f"\n[MSE] Final Val MSE: {mse_val2:.6f} | Val Pearson: {r_val2:.4f}")

    # Optional: print a tiny JSON summary to copy into notes
    summary = {
        "mixed": {"val_mse": mse_val,  "val_pearson": r_val,  "lambda_prop": LAMBDA_PROP},
        "mse":   {"val_mse": mse_val2, "val_pearson": r_val2}
    }
    print("\nSummary:", json.dumps(summary, indent=2))
