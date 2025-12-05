#!/usr/bin/env python3
"""
Shallow Transformer Random Search (GPU + AMP)

This script trains shallow transformer models to predict isoform 
expression from gene expression. It performs a random search over:

- Patch sizes (groups of genes per "token")
- Transformer d_model, num_heads, num_layers
- Pooling modes ("mean", "attn")
- Activations ("gelu", "relu")
- Dropout
- Learning rates
- Batch sizes

What is kept identical to FFNN script:
- Data loading (Xg_log1p, log1p(Y_tx))
- Train/val/test split (same SEED + sklearn train_test_split)
- Normalization (train-mean / train-std on X, log1p on Y)
- Loss (MSE in log space)
- Early stopping on validation MSE
- Final test MSE and Pearson correlation
"""

import os, json, math, random, csv, time, pathlib
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

N_TRIALS        = 10
MAX_EPOCHS      = 120
PATIENCE        = 10
EVAL_EVERY      = 5
GRAD_CLIP       = 1.0

# HPO search ranges
BATCHES         = [8, 16]                # larger batch uses more GPU memory
LRS             = [1e-3, 2e-3, 3e-3]
DROPOUTS        = [0.0, 0.1, 0.2]

PATCH_SIZES     = [64, 128, 256]           # genes per token
DMODELS         = [64, 96, 128]  # transformer embedding dims
N_HEADS_OPTIONS = [2, 4]               # must divide d_model
N_LAYERS        = [1, 2]                  # transformer layers
POOLINGS        = ["mean", "attn"]
ACTIVATIONS     = ["gelu", "relu"]

RESULTS_CSV      = "results_v1/arch_search_results.csv"
BEST_MODEL_PT    = "results_v1/best_model.pt"
SUMMARY_JSON     = "results_v1/arch_search_summary.json"

os.environ.setdefault("MPLBACKEND", "Agg")
TRIAL_FIG_DIR    = "results_v1/figs_trials"
SUMMARY_FIG_BAR  = "results_v1/summary_val_mse_bar.png"
SUMMARY_FIG_TOP5 = "results_v1/top5_val_curves.png"

# ------------------------------
# Repro & matmul knobs
# ------------------------------
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ------------------------------
# Data path (portable)
# ------------------------------
DATA_PT = os.environ.get("DATA_PT")

if DATA_PT is None:
    # Fall back to $BLACKHOLE/$USER/data.pt
    bh   = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export DATA_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    DATA_PT = os.path.join(bh, user, "data.pt")

print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

# X: gene expression (already log1p in your setup)
# Y: isoform expression (we add log1p here)
X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

N, G = X.shape
_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}, Y={Y.shape}")

# ------------------------------
# Split: train / val / test
# (same logic & SEED as FFNN script)
# ------------------------------
all_idx   = np.arange(N)
trval_idx, te_idx = train_test_split(
    all_idx,
    test_size=TEST_FRAC,
    random_state=SEED,
    shuffle=True
)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(
    trval_idx,
    test_size=val_rel,
    random_state=SEED,
    shuffle=True
)
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ------------------------------
# Normalize once (train stats) + device tensors
# ------------------------------
X_mean = X[tr_idx].mean(axis=0)
X_std  = X[tr_idx].std(axis=0) + 1e-8
Xz     = (X - X_mean) / X_std

Xt = torch.from_numpy(Xz).to(DEVICE).float()
Yt = torch.from_numpy(Y ).to(DEVICE).float()

tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
te_idx_t = torch.from_numpy(te_idx).to(DEVICE)

def batch_iter(idxs_t: torch.Tensor, batch_size: int, shuffle: bool = True):
    if shuffle:
        idxs_t = idxs_t[torch.randperm(idxs_t.numel(), device=idxs_t.device)]
    for i in range(0, idxs_t.numel(), batch_size):
        j = idxs_t[i:i+batch_size]
        yield Xt.index_select(0, j), Yt.index_select(0, j)

# ------------------------------
# Loss & metrics
# ------------------------------
criterion = nn.MSELoss()

def pearson_mean_gpu(Y_true: torch.Tensor, Y_pred: torch.Tensor) -> float:
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
# Model: Shallow Patch Transformer with pooling & activation hyperparams
# ------------------------------
class ShallowPatchTransformer(nn.Module):
    """
    Treats genes as a 1D sequence of patches:

      - Input: x ∈ R^{batch, G}
      - Split into patches of size P: tokens ∈ R^{batch, n_patches, P}
      - Linear patch embedding to d_model
      - Add learned positional encodings
      - 1–2 TransformerEncoder layers
      - Pool patches into a sample-level vector (mean or attention pooling)
      - MLP head to I isoform outputs
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        pooling: str = "mean",    # "mean" or "attn"
        activation: str = "gelu", # "gelu" or "relu"
    ):
        super().__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.patch_size = patch_size
        self.pooling    = pooling.lower()
        self.activation = activation.lower()

        # Number of patches (ceil in case G is not divisible by patch_size)
        self.n_patches = math.ceil(in_dim / patch_size)

        # Patch embedding: (B, n_patches, P) -> (B, n_patches, d_model)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model          = d_model,
            nhead            = n_heads,
            dim_feedforward  = 4 * d_model,
            dropout          = dropout,
            activation       = self.activation,  # "gelu" or "relu"
            batch_first      = True,
            norm_first       = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling head (for pooling="attn")
        if self.pooling == "attn":
            self.attn_pool = nn.Linear(d_model, 1)

        # Choose activation layer for the head
        if self.activation == "relu":
            act_layer = nn.ReLU()
        else:
            act_layer = nn.GELU()

        # MLP head
        mlp_hidden = max(128, d_model * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, G)
        """
        B, D = x.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got {D}"

        # Pad to multiple of patch_size if needed
        pad_len = self.n_patches * self.patch_size - D
        if pad_len > 0:
            pad = x.new_zeros(B, pad_len)
            x = torch.cat([x, pad], dim=1)  # (B, n_patches * patch_size)

        # Reshape into patches
        x = x.view(B, self.n_patches, self.patch_size)   # (B, n_patches, patch_size)

        # Patch embedding + positional encodings
        x = self.patch_embed(x)                          # (B, n_patches, d_model)
        x = x + self.pos_embed                           # (B, n_patches, d_model)

        # Transformer encoder
        x = self.encoder(x)                              # (B, n_patches, d_model)

        # ---- POOLING ----
        if self.pooling == "mean":
            # Global mean pooling over patches
            pooled = x.mean(dim=1)                       # (B, d_model)

        elif self.pooling == "attn":
            # Attention pooling over patches
            scores = self.attn_pool(x)                   # (B, n_patches, 1)
            alpha  = torch.softmax(scores, dim=1)        # (B, n_patches, 1)
            pooled = (alpha * x).sum(dim=1)              # (B, d_model)

        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        # Head to outputs
        out = self.head(pooled)                          # (B, out_dim)
        return out

# ------------------------------
# Evaluation helper
# ------------------------------
def evaluate_on(model: nn.Module, idxs_t: torch.Tensor, batch_size: int) -> float:
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

def evaluate_pearson_on(model: nn.Module, idxs_t: torch.Tensor, batch_size: int) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in batch_iter(idxs_t, batch_size=batch_size, shuffle=False):
            with torch.cuda.amp.autocast(enabled=AMP):
                preds.append(model(xb))
    Y_true = Yt[idxs_t]
    Y_pred = torch.cat(preds, dim=0)
    return pearson_mean_gpu(Y_true, Y_pred)

# ------------------------------
# One training run (per trial)
# ------------------------------
def train_once(hp: dict, trial_seed: int):
    """
    hp keys:
      - name, patch_size, d_model, n_heads, num_layers
      - pooling, activation, dropout, lr, batch_size, epochs
    """
    set_seed(trial_seed)

    model = ShallowPatchTransformer(
        in_dim     = G,
        out_dim    = I,
        patch_size = hp["patch_size"],
        d_model    = hp["d_model"],
        n_heads    = hp["n_heads"],
        num_layers = hp["num_layers"],
        dropout    = hp["dropout"],
        pooling    = hp["pooling"],
        activation = hp["activation"],
    ).to(DEVICE)

    opt    = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sch    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_val   = math.inf
    best_state = None
    noimp      = 0
    train_curve, val_curve = [], []

    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in batch_iter(tr_idx_t, batch_size=hp["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                pb   = model(xb)
                loss = criterion(pb, yb)
            scaler.scale(loss).backward()
            if GRAD_CLIP is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * yb.size(0)
            seen  += yb.size(0)

        epoch_train = total / max(1, seen)
        train_curve.append(epoch_train)

        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_mse = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
            val_r   = evaluate_pearson_on(model, va_idx_t, batch_size=hp["batch_size"])

            val_curve.append(val_mse)
            sch.step(val_mse)

            print(
                f"[{hp['name']}] ep {epoch:03d} | "
                f"patch={hp['patch_size']} d_model={hp['d_model']} "
                f"heads={hp['n_heads']} layers={hp['num_layers']} "
                f"pool={hp['pooling']} act={hp['activation']} | "
                f"train {epoch_train:.5f} | val {val_mse:.5f} | "
                f"r {val_r:.4f} | lr {opt.param_groups[0]['lr']:.4g}"
            )


            if val_mse < best_val - 0.0:
                best_val   = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp      = 0
            else:
                noimp += 1

            if noimp >= PATIENCE:
                print(f"[{hp['name']}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse     = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
    train_time  = time.time() - t0

    # Per-trial plot
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{hp['name']}.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(train_curve, label="train MSE")
        xs_v = [e for e in range(1, len(train_curve)+1)
                if e == 1 or e % EVAL_EVERY == 0][:len(val_curve)]
        plt.plot(xs_v, val_curve, "o-", label="val MSE")
        plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.title(
            f"{hp['name']} — p={hp['patch_size']} d={hp['d_model']} "
            f"h={hp['n_heads']} L={hp['num_layers']} "
            f"pool={hp['pooling']} act={hp['activation']}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plot failed for {hp['name']}: {e}")

    rec = {
        "name":           hp["name"],
        "patch_size":     hp["patch_size"],
        "d_model":        hp["d_model"],
        "n_heads":        hp["n_heads"],
        "num_layers":     hp["num_layers"],
        "pooling":        hp["pooling"],
        "activation":     hp["activation"],
        "dropout":        hp["dropout"],
        "lr":             hp["lr"],
        "momentum":       "-",
        "step_size":      "-",
        "gamma":          "-",
        "batch_size":     hp["batch_size"],
        "epochs_trained": len(train_curve),
        "val_mse":        float(val_mse),
        "val_pearson": float(val_r),
        "train_time_sec": round(train_time, 1),
    }
    curves = {"train": train_curve, "val": val_curve}
    return rec, model, curves

# ------------------------------
# Search space
# ------------------------------
def sample_hp(t: int) -> dict:
    rnd = random.Random(SEED + 2000 + t)

    # Choose a valid (d_model, n_heads) with divisibility
    while True:
        d_model = rnd.choice(DMODELS)
        n_heads = rnd.choice(N_HEADS_OPTIONS)
        if d_model % n_heads == 0:
            break

    patch      = rnd.choice(PATCH_SIZES)
    n_layers   = rnd.choice(N_LAYERS)
    pooling    = rnd.choice(POOLINGS)
    activation = rnd.choice(ACTIVATIONS)

    return {
        "name":       f"tf_p{patch}_d{d_model}_h{n_heads}_L{n_layers}_{pooling}_{activation}_t{t}",
        "patch_size": patch,
        "d_model":    d_model,
        "n_heads":    n_heads,
        "num_layers": n_layers,
        "pooling":    pooling,
        "activation": activation,
        "dropout":    rnd.choice(DROPOUTS),
        "lr":         rnd.choice(LRS),
        "batch_size": rnd.choice(BATCHES),
        "epochs":     MAX_EPOCHS,
    }

# ------------------------------
# Run search
# ------------------------------
print(f"[INFO] Starting shallow transformer random search with {N_TRIALS} trials...")
results, curves_by_name = [], {}
best_rec, best_model, best_hp = None, None, None

pathlib.Path(os.path.dirname(RESULTS_CSV)).mkdir(parents=True, exist_ok=True)

fieldnames = [
    "name","patch_size","d_model","n_heads","num_layers","pooling","activation",
    "dropout","lr","momentum","step_size","gamma","batch_size",
    "epochs_trained","val_mse","val_pearson","train_time_sec"
]

# CSV header
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

for t in range(N_TRIALS):
    hp = sample_hp(t)
    print(f"\n=== Trial {t+1}/{N_TRIALS}: {hp} ===")
    rec, model, curves = train_once(hp, trial_seed=SEED + t)

    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(rec)

    results.append(rec)
    curves_by_name[rec["name"]] = curves

    if best_rec is None or rec["val_mse"] < best_rec["val_mse"]:
        best_rec   = rec
        best_model = model
        best_hp    = hp
        pathlib.Path(os.path.dirname(BEST_MODEL_PT)).mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_model.state_dict(),
            "hparams":    {k: v for k, v in hp.items()},
            "X_mean":     X_mean,
            "X_std":      X_std,
            "meta": {
                "G": G,
                "I": I,
                "seed": SEED,
                "model_type": "shallow_patch_transformer",
            },
        }, BEST_MODEL_PT)
        print(f"[BEST] Updated best by Val: {rec['name']} (val_mse={rec['val_mse']:.6f})")

# ------------------------------
# Test only once (best model) — compute Pearson here
# ------------------------------
print("\n[INFO] Evaluating test set for the best-by-val model only...")
best_bs = best_hp["batch_size"]
best_model.eval()

preds = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=best_bs, shuffle=False):
        with torch.cuda.amp.autocast(enabled=AMP):
            preds.append(best_model(xb))

Y_test_t = Yt[te_idx_t]           # tensor on GPU
Y_pred_t = torch.cat(preds, dim=0)

test_mse = float(((Y_test_t - Y_pred_t)**2).mean().item())
test_r   = pearson_mean_gpu(Y_test_t, Y_pred_t)

best_rec["test_mse"]     = test_mse
best_rec["test_pearson"] = test_r

print(
    f"[BEST on TEST] patch={best_hp['patch_size']} d_model={best_hp['d_model']} "
    f"heads={best_hp['n_heads']} layers={best_hp['num_layers']} "
    f"pool={best_hp['pooling']} act={best_hp['activation']} | "
    f"MSE: {test_mse:.6f} | r: {test_r:.4f}"
)

# ------------------------------
# Summary files + summary plots
# ------------------------------
results_sorted = sorted(results, key=lambda r: r["val_mse"])
summary = {
    "best":          best_rec,
    "top5":          results_sorted[:5],
    "n_trials":      N_TRIALS,
    "csv":           RESULTS_CSV,
    "best_model_pt": BEST_MODEL_PT,
}
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SEARCH COMPLETE ===")
print(json.dumps(summary, indent=2))

# Bar chart of Val MSE (all models)
try:
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results_sorted]
    vals  = [r["val_mse"] for r in results_sorted]
    plt.figure(figsize=(max(8, 0.4*len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title("Shallow Transformer — Validation MSE by model")
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_BAR, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot summary bar: {e}")

# Overlay of top-5 validation curves
try:
    import matplotlib.pyplot as plt
    top5 = results_sorted[:5]
    plt.figure(figsize=(8, 5))
    for r in top5:
        c = curves_by_name[r["name"]]
        xs_v = [e for e in range(1, len(c["train"])+1)
                if e == 1 or e % EVAL_EVERY == 0][:len(c["val"])]
        plt.plot(xs_v, c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        plt.plot(range(1, len(c["train"])+1), c["train"], alpha=0.4, linewidth=1.0)
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Shallow Transformer — Top-5 models (validation curves)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot top-5 overlay: {e}")

print("\n=== BEST MODEL HYPERPARAMETERS ===")
for k, v in best_hp.items():
    print(f"{k}: {v}")

print("[INFO] All done.")
