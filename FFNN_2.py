#!/usr/bin/env python3
"""
FFNN architecture & hyperparameter search (train/val/test, simple plots)

What it does
------------
- Loads preprocessed tensors from DATA_PT (env) or ./data.pt
  expects: Xg_log1p (N,G), Y_tx (N,I); Y is log1p'ed here.
- He/Kaiming init (ReLU), SGD+momentum, StepLR.
- Train/Val/Test split.
- Random search over several architectures/hparams.
- For each trial: saves a figure with Train/Val MSE curves (no test leak).
- Picks best model by Val MSE, then evaluates Test once.
- Saves:
    - arch_search_results.csv
    - arch_search_summary.json
    - best_model.pt (weights + hparams + norm stats)
    - figs_trials/curves_<trial>.png
    - summary_val_mse_bar.png
    - top5_val_curves.png
    - best_model_curves.png   (includes test line for the final model)
"""

import os, json, math, random, csv, time, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

N_TRIALS        = 12          # raise for deeper search
MAX_EPOCHS      = 120
PATIENCE        = 15
GRAD_CLIP       = 1.0
BATCH_SIZE_DEF  = 16          # mind output size (I is large)

RESULTS_CSV     = "arch_search_results.csv"
BEST_MODEL_PT   = "best_model.pt"
SUMMARY_JSON    = "arch_search_summary.json"

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting
TRIAL_FIG_DIR    = "figs_trials"
SUMMARY_FIG_BAR  = "summary_val_mse_bar.png"
SUMMARY_FIG_TOP5 = "top5_val_curves.png"
BEST_FIG_CURVES  = "best_model_curves.png"

# ------------------------------
# Repro
# ------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ------------------------------
# Data
# ------------------------------
DATA_PT = os.environ.get("DATA_PT", "data.pt")
print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu")

# X already log1p; Y -> log1p here
X = data["Xg_log1p"].numpy().astype(np.float32)       # (N, G)
Y = np.log1p(data["Y_tx"].numpy()).astype(np.float32) # (N, I)

N, G = X.shape
_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}, Y={Y.shape}")

# ------------------------------
# Split: train / val / test
# ------------------------------
all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel, random_state=SEED, shuffle=True)
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ------------------------------
# Normalization (train stats)
# ------------------------------
X_mean = X[tr_idx].mean(axis=0)
X_std  = X[tr_idx].std(axis=0) + 1e-8

def batch_iter(idxs, batch_size=BATCH_SIZE_DEF, shuffle=True):
    if shuffle:
        order = np.random.permutation(idxs)
    else:
        order = np.array(idxs, copy=True)
    for i in range(0, len(order), batch_size):
        j = order[i:i+batch_size]
        xb = (X[j] - X_mean) / X_std
        yb = Y[j]
        yield torch.from_numpy(xb).to(DEVICE), torch.from_numpy(yb).to(DEVICE)

# ------------------------------
# Model
# ------------------------------
class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, act="relu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU() if act=="relu" else nn.GELU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ------------------------------
# Metrics / Eval
# ------------------------------
criterion = nn.MSELoss()

def evaluate_on(model, idxs, batch_size):
    model.eval()
    loss_sum, nsamp = 0.0, 0
    preds = []
    with torch.no_grad():
        for xb, yb in batch_iter(idxs, batch_size=batch_size, shuffle=False):
            pb = model(xb)
            loss_sum += float(criterion(pb, yb).item()) * yb.shape[0]
            nsamp += yb.shape[0]
            preds.append(pb.detach().cpu())
    Yref = torch.from_numpy(Y[idxs])
    P = torch.cat(preds, dim=0)
    yt = Yref - Yref.mean(dim=0, keepdim=True)
    yp = P    - P.mean(dim=0, keepdim=True)
    denom = (torch.sqrt((yt**2).sum(dim=0)) * torch.sqrt((yp**2).sum(dim=0)) + 1e-8)
    r = (yt * yp).sum(dim=0) / denom
    pearson = float(torch.nanmean(r).item())
    return loss_sum / max(1, nsamp), pearson

# ------------------------------
# One training run (no test here)
# ------------------------------
def train_once(hp, trial_seed):
    """
    hp keys:
      - name, hidden, dropout, batchnorm, lr, momentum, step_size, gamma, batch_size, epochs
    """
    set_seed(trial_seed)

    model = FFNN(G, I,
                 hidden=hp["hidden"],
                 act="relu",
                 dropout=hp["dropout"],
                 batchnorm=hp["batchnorm"]).to(DEVICE)
    model.apply(he_init)

    opt = torch.optim.SGD(model.parameters(), lr=hp["lr"], momentum=hp["momentum"])
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=hp["step_size"], gamma=hp["gamma"])

    best_val = math.inf
    best_state = None
    noimp = 0
    train_curve, val_curve = [], []

    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        train_sum, n_seen = 0.0, 0
        for xb, yb in batch_iter(tr_idx, batch_size=hp["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            pb = model(xb)
            loss = criterion(pb, yb)
            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            train_sum += float(loss.item()) * yb.shape[0]
            n_seen += yb.shape[0]
        epoch_train = train_sum / max(1, n_seen)

        val_mse, val_r = evaluate_on(model, va_idx, batch_size=hp["batch_size"])
        sch.step()

        train_curve.append(epoch_train)
        val_curve.append(val_mse)

        if val_mse < best_val - 0.0:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"[{hp['name']}] Epoch {epoch:03d} | Train MSE: {epoch_train:.6f} | Val MSE: {val_mse:.6f} | Val r: {val_r:.4f} | LR: {opt.param_groups[0]['lr']:.5f}")

        if noimp >= PATIENCE:
            print(f"[{hp['name']}] Early stopping.")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Final val metrics (post-restore)
    val_mse, val_r = evaluate_on(model, va_idx, batch_size=hp["batch_size"])
    train_time = time.time() - t0

    # Per-trial plot (train/val only; no test line here to avoid leakage)
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{hp['name']}.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(train_curve, label="train MSE")
        plt.plot(val_curve, label="val MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title(f"Training curves — {hp['name']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plot failed for {hp['name']}: {e}")

    rec = {
        "name": hp["name"],
        "hidden": hp["hidden"],
        "dropout": hp["dropout"],
        "batchnorm": hp["batchnorm"],
        "lr": hp["lr"],
        "momentum": hp["momentum"],
        "step_size": hp["step_size"],
        "gamma": hp["gamma"],
        "batch_size": hp["batch_size"],
        "epochs_trained": len(train_curve),
        "val_mse": float(val_mse),
        "val_pearson": float(val_r),
        "train_time_sec": round(train_time, 1),
    }
    curves = {"train": train_curve, "val": val_curve}
    return rec, model, curves

# ------------------------------
# Search space
# ------------------------------
ARCHS = [
    ("shallow_1024",        [1024]),
    ("deep_2048_1024",      [2048, 1024]),
    ("narrow_x4_512",       [512, 512, 512, 512]),
    ("bottleneck_2048_256", [2048, 256, 2048]),
    ("medium_1536_768_384", [1536, 768, 384]),
    ("small_256_256",       [256, 256]),
]
DROPOUTS   = [0.0, 0.1, 0.2]
BATCHNORMS = [False, True]
LRS        = [0.1, 0.05, 0.02]
MOMS       = [0.9]
STEP_SIZES = [10, 15]
GAMMAS     = [0.5, 0.3]
BATCHES    = [8, 16, 24]

def sample_hp(t):
    rnd = random.Random(SEED + 1000 + t)
    name, hidden = rnd.choice(ARCHS)
    return {
        "name": f"{name}_t{t}",
        "hidden": hidden,
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
        "momentum": rnd.choice(MOMS),
        "step_size": rnd.choice(STEP_SIZES),
        "gamma": rnd.choice(GAMMAS),
        "batch_size": rnd.choice(BATCHES),
        "epochs": MAX_EPOCHS,
    }

# ------------------------------
# Run search
# ------------------------------
print(f"[INFO] Starting random search with {N_TRIALS} trials...")
results, curves_by_name = [], {}
best_rec, best_model, best_hp = None, None, None

# CSV header
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name","hidden","dropout","batchnorm","lr","momentum","step_size","gamma","batch_size",
        "epochs_trained","val_mse","val_pearson","train_time_sec"
    ])
    w.writeheader()

for t in range(N_TRIALS):
    hp = sample_hp(t)
    print(f"\n=== Trial {t+1}/{N_TRIALS}: {hp} ===")
    rec, model, curves = train_once(hp, trial_seed=SEED + t)

    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","hidden","dropout","batchnorm","lr","momentum","step_size","gamma","batch_size",
            "epochs_trained","val_mse","val_pearson","train_time_sec"
        ])
        w.writerow(rec)

    results.append(rec)
    curves_by_name[rec["name"]] = curves

    if best_rec is None or rec["val_mse"] < best_rec["val_mse"]:
        best_rec = rec
        best_model = model
        best_hp = hp
        torch.save({
            "state_dict": best_model.state_dict(),
            "hparams": {k: v for k, v in hp.items()},
            "X_mean": X_mean, "X_std": X_std,
            "meta": {"G": G, "I": I, "seed": SEED}
        }, BEST_MODEL_PT)
        print(f"[BEST] Updated best by Val: {rec['name']} (val_mse={rec['val_mse']:.6f})")

# ------------------------------
# Test only once (best model)
# ------------------------------
print("\n[INFO] Evaluating test set for the best-by-val model only...")
test_mse, test_r = evaluate_on(best_model, te_idx, batch_size=best_hp["batch_size"])
best_rec["test_mse"] = float(test_mse)
best_rec["test_pearson"] = float(test_r)
print(f"[BEST on TEST] MSE: {test_mse:.6f} | r: {test_r:.4f}")

# Save a best-model curves plot including the test horizontal line
try:
    import matplotlib.pyplot as plt
    c = curves_by_name[best_rec["name"]]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(c["train"], label="train MSE")
    plt.plot(c["val"], label="val MSE")
    plt.axhline(test_mse, linestyle="--", linewidth=1.2, label=f"test MSE = {test_mse:.4g}")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title(f"Best model — {best_rec['name']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BEST_FIG_CURVES, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Plot best curves failed: {e}")

# ------------------------------
# Summary files + summary plots
# ------------------------------
results_sorted = sorted(results, key=lambda r: r["val_mse"])
summary = {
    "best": best_rec,
    "top5": results_sorted[:5],
    "n_trials": N_TRIALS,
    "csv": RESULTS_CSV,
    "best_model_pt": BEST_MODEL_PT
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
    plt.title("Validation MSE by model")
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_BAR, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot summary bar: {e}")

# Overlay of top-5 validation curves (train faint), no test lines here
try:
    import matplotlib.pyplot as plt
    top5 = results_sorted[:5]
    plt.figure(figsize=(8, 5))
    for r in top5:
        c = curves_by_name[r["name"]]
        plt.plot(c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        plt.plot(c["train"], alpha=0.4, linewidth=1.0)
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Top-5 models — validation curves")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot top-5 overlay: {e}")
print("[INFO] All done.")