#!/usr/bin/env python3
"""
Simple ResidualMLP training on gene -> isoform prediction.

- Loads the same data as the FFNN script
- Builds a single ResidualMLP
- Trains with fixed hyperparameters
- Prints train/val/test MSE and Pearson

This is meant just to verify that a residual architecture runs end-to-end.
"""

import os, math, time, random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AMP             = (DEVICE == "cuda")

# Fixed training hyperparameters (simple test)
MAX_EPOCHS      = 60    # number of training epochs
BATCH_SIZE      = 192   # number of samples per batch
LR              = 1e-3  # how big each parameter update is (standard 1e-3 for AdamW)
GRAD_CLIP       = 1.0   # max norm for gradient clipping, avoid exploding gradients

DROPOUT         = 0.1   # randomly sets 10% of activations to zero in each residual block
HIDDEN_DIM      = 512   # embedding dimension
NUM_BLOCKS      = 4     # number of residual blocks

# ------------------------------
# Repro & matmul knobs
# ------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ------------------------------
# Data path (portable)
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
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

# --- Y: isoform expression ---
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

# --- X: now comes from VAE latents, not Xg_log1p ---
LATENT_PT = os.environ.get("LATENT_PT")
if LATENT_PT is None:
    # default: same dir as DATA_PT, file name vae_latents.pt
    data_dir = os.path.dirname(DATA_PT)
    LATENT_PT = os.path.join(data_dir, "vae_latents_all.pt")

print(f"[INFO] Loading VAE latents from: {LATENT_PT}")
latents = torch.load(LATENT_PT, map_location="cpu")

if "Z" not in latents:
    raise KeyError(
        "vae_latents.pt must contain key 'Z' with shape (N, latent_dim). "
        f"Got keys: {list(latents.keys())}"
    )

Z = latents["Z"].float().cpu().numpy()   # shape (N, latent_dim)
X = Z                                    # treat latents as inputs

N, G = X.shape
_, I = Y.shape
print(f"[INFO] Shapes: X(latents)={X.shape}, Y={Y.shape}")

# ------------------------------
# Split: train / val / test
# ------------------------------
all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel, random_state=SEED, shuffle=True)
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ------------------------------
# Normalize once (train stats) + device tensors
# (now normalization is applied to latent features instead of genes)
# ------------------------------
X_mean = X[tr_idx].mean(axis=0)
X_std  = X[tr_idx].std(axis=0) + 1e-8
Xz = (X - X_mean) / X_std

Xt = torch.from_numpy(Xz).to(DEVICE).float()
Yt = torch.from_numpy(Y ).to(DEVICE).float()

tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
te_idx_t = torch.from_numpy(te_idx).to(DEVICE)

def batch_iter(idxs_t, batch_size, shuffle=True):
    if shuffle:
        idxs_t = idxs_t[torch.randperm(idxs_t.numel(), device=idxs_t.device)]
    for i in range(0, idxs_t.numel(), batch_size):
        j = idxs_t[i:i+batch_size]
        yield Xt.index_select(0, j), Yt.index_select(0, j)

# ------------------------------
# Residual MLP definition
# ------------------------------
class ResidualBlock(nn.Module):
    """
    A single pre-activation residual MLP block:

    x -----> LayerNorm -> Activation -> Linear (dim -> hidden_dim)
              -> Activation -> Linear (hidden_dim -> dim)
              -> Dropout -> + x (skip connection)
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        # Normalize across the feature dimension (genes/embedding)
        self.norm = nn.LayerNorm(dim)

        # Two-layer MLP inside the residual branch
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.GELU()  # You can swap for nn.ReLU() if you prefer

        # Initialization:
        # - He init for the first layer (good for ReLU/GELU)
        # - Very small weights for the second layer so the block starts close to identity
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save the input for the skip connection
        residual = x

        # Pre-activation: norm -> activation -> MLP
        out = self.norm(x)
        out = self.activation(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)

        # Skip connection: output = input + residual branch
        return residual + out


class ResidualMLP(nn.Module):
    """
    Residual MLP for gene -> isoform prediction.

    Architecture:
        - Input: normalized gene expression (num_genes)
        - Linear layer to go from input_dim (num genes) to hidden_dim
        - Several ResidualBlocks at dimension hidden_dim
        - Output linear layer to num_isoforms
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # First layer: project raw gene expression to a lower-dimensional embedding
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Stack of residual blocks operating at dimension hidden_dim
        self.blocks = nn.ModuleList([
            ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim * 2, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Final prediction layer: from embedding to isoform expression
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize input and output layers
        self._init_weights()

    def _init_weights(self):
        # He init for layers followed by non-linearities
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_layer.bias)

        # Small init for the output layer to keep early predictions near zero
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch_size, num_genes)
           containing normalized gene expression.
        returns: tensor of shape (batch_size, num_isoforms)
        """
        # Project genes to hidden representation
        h = self.input_layer(x)

        # Pass through each residual block
        for block in self.blocks:
            h = block(h)

        # Final linear layer to isoform expression
        y = self.output_layer(h)
        return y

# ------------------------------
# Metrics / Eval
# ------------------------------
criterion = nn.MSELoss()

def evaluate_on(model, idxs_t, batch_size):
    """
    Compute MSE over a given index set (train/val/test).
    """
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for xb, yb in batch_iter(idxs_t, batch_size=batch_size, shuffle=False):
            with torch.cuda.amp.autocast(enabled=AMP):
                pb = model(xb)
                loss = criterion(pb, yb)
            total += float(loss.item()) * yb.size(0)
            seen  += yb.size(0)
    return total / max(1, seen)

def pearson_mean_gpu(Y_true, Y_pred):
    """
    Fast GPU Pearson correlation (mean over outputs).
    Y_true, Y_pred: (N, I) tensors on GPU
    """
    yt = Y_true - Y_true.mean(dim=0, keepdim=True)
    yp = Y_pred - Y_pred.mean(dim=0, keepdim=True)

    num = (yt * yp).sum(dim=0)
    den = torch.sqrt((yt * yt).sum(dim=0)) * torch.sqrt((yp * yp).sum(dim=0)) + 1e-8

    r = num / den
    return r.nanmean().item()

# ------------------------------
# Build model + optimizer
# ------------------------------
print("[INFO] Building ResidualMLP model...")
model = ResidualMLP(
    input_dim=G,
    hidden_dim=HIDDEN_DIM,
    num_blocks=NUM_BLOCKS,
    output_dim=I,
    dropout=DROPOUT,
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler('cuda', enabled=AMP)

# ------------------------------
# Simple training loop
# ------------------------------
print("[INFO] Starting training...")
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    total, seen = 0.0, 0

    for xb, yb in batch_iter(tr_idx_t, batch_size=BATCH_SIZE, shuffle=True):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=AMP):
            pb = model(xb)
            loss = criterion(pb, yb)
        scaler.scale(loss).backward()

        if GRAD_CLIP is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(opt)
        scaler.update()

        total += float(loss.item()) * yb.size(0)
        seen  += yb.size(0)

    train_mse = total / max(1, seen)
    val_mse   = evaluate_on(model, va_idx_t, batch_size=BATCH_SIZE)

    print(f"[EPOCH {epoch:03d}] train MSE={train_mse:.6f} | val MSE={val_mse:.6f}")

train_time = time.time() - t0
print(f"[INFO] Training finished in {train_time/60:.1f} min")

# ------------------------------
# Final evaluation on train/val/test
# ------------------------------
print("\n[INFO] Final evaluation...")

train_mse = evaluate_on(model, tr_idx_t, batch_size=BATCH_SIZE)
val_mse   = evaluate_on(model, va_idx_t, batch_size=BATCH_SIZE)
test_mse  = evaluate_on(model, te_idx_t, batch_size=BATCH_SIZE)

print(f"Train MSE: {train_mse:.6f}")
print(f"Val   MSE: {val_mse:.6f}")
print(f"Test  MSE: {test_mse:.6f}")

# Pearson on test set
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=BATCH_SIZE, shuffle=False):
        with torch.cuda.amp.autocast(enabled=AMP):
            preds.append(model(xb))

Y_test_t = Yt[te_idx_t]
Y_pred_t = torch.cat(preds, dim=0)

test_r = pearson_mean_gpu(Y_test_t, Y_pred_t)
print(f"Test Pearson r (mean over isoforms): {test_r:.4f}")

print("\n[INFO] Done.")
