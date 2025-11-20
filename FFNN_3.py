#!/usr/bin/env python3
"""
FFNN architecture & hyperparameter search (FAST, activation sweep)
- Tests activations: relu, gelu, tanh, leakyrelu (cycled across trials)
- AdamW + AMP (new torch.amp API), safe on CPU/GPU
- Large batches, normalized once, early stopping
- Full validation and logging every 10 epochs (set EVAL_EVERY=1 for every-epoch)
- Saves:
    arch_search_results.csv
    arch_search_summary.json
    best_model.pt
    figs_trials/curves_<trial>.png
    summary_val_mse_bar.png
    top5_val_curves.png
    best_model_curves.png
"""

import os, json, math, random, csv, time, pathlib, contextlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ------------------------------
# Config
# ------------------------------
SEED         = 42
TEST_FRAC    = 0.15
VAL_FRAC     = 0.15

# Log/eval cadence (adjust as you like)
LOG_EVERY    = 10           # print every 10 epochs
EVAL_EVERY   = 1            # full validation every 10 epochs (and at epoch 1)
PATIENCE     = 6            # early stop checks only on evaluation epochs

# Search space (beefed up LR; tune if unstable)
N_TRIALS     = 12
MAX_EPOCHS   = 120
GRAD_CLIP    = 1.0
BATCHES      = [192, 256, 320]           # fit to your RAM/GPU
LRS          = [3e-3, 1e-2, 3e-2]        # larger to speed convergence
DROPOUTS     = [0.0, 0.1, 0.2]
BATCHNORMS   = [False, True]
ACTS         = ["relu", "gelu", "tanh", "leakyrelu"]  # explicit cycle

# IO
RESULTS_CSV      = "arch_search_results.csv"
BEST_MODEL_PT    = "best_model.pt"
SUMMARY_JSON     = "arch_search_summary.json"
TRIAL_FIG_DIR    = "figs_trials"
SUMMARY_FIG_BAR  = "summary_val_mse_bar.png"
SUMMARY_FIG_TOP5 = "top5_val_curves.png"
BEST_FIG_CURVES  = "best_model_curves.png"

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting

# ------------------------------
# Device & AMP (new API)
# ------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLE   = True
USE_CUDA_AMP = AMP_ENABLE and DEVICE.type == "cuda"
USE_CPU_AMP  = AMP_ENABLE and DEVICE.type == "cpu" and torch.backends.cpu.is_bf16_supported()

SCALER = torch.amp.GradScaler('cuda', enabled=USE_CUDA_AMP)

def autocast_ctx():
    if USE_CUDA_AMP:
        return torch.amp.autocast('cuda', dtype=torch.float16)
    elif USE_CPU_AMP:
        return torch.amp.autocast('cpu', dtype=torch.bfloat16)
    else:
        return contextlib.nullcontext()

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True  # good speedup on Ampere+

# ------------------------------
# Repro
# ------------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ------------------------------
# Data
# ------------------------------
DATA_PT = os.environ.get("DATA_PT", "data.pt")
print(f"[INFO] Loading tensors from: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu")

# X already in log1p; Y -> log1p here (matches previous pipeline)
X = data["Xg_log1p"].numpy().astype(np.float32)        # (N, G)
Y = np.log1p(data["Y_tx"].numpy()).astype(np.float32)  # (N, I)
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
# Normalize once (train stats), move to device
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
# Activations & init
# ------------------------------
def get_activation(name: str):
    n = name.lower()
    if n == "relu": return nn.ReLU()
    if n == "gelu": return nn.GELU()
    if n == "tanh": return nn.Tanh()
    if n in ("leakyrelu", "leaky_relu"): return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation: {name}")

def init_linear(m: nn.Module, act: str):
    if not isinstance(m, nn.Linear): return
    if act.lower() == "tanh":
        nn.init.xavier_normal_(m.weight)
    else:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
    if m.bias is not None:
        nn.init.zeros_(m.bias)

# ------------------------------
# Model
# ------------------------------
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
        self.act_name = act
    def forward(self, x): return self.net(x)

# ------------------------------
# Metrics / Eval
# ------------------------------
criterion = nn.MSELoss()

def evaluate_on(model, idxs_t, batch_size):
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for xb, yb in batch_iter(idxs_t, batch_size=batch_size, shuffle=False):
            with autocast_ctx():
                pb = model(xb)
                loss = criterion(pb, yb)
            total += float(loss.item()) * yb.size(0)
            seen  += yb.size(0)
    return total / max(1, seen)

def pearson_mean_np(a, b):
    A = torch.from_numpy(a); B = torch.from_numpy(b)
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    r = (A*B).sum(dim=0) / (torch.sqrt((A*A).sum(dim=0)) * torch.sqrt((B*B).sum(dim=0)) + 1e-8)
    return float(torch.nanmean(r).item())

# ------------------------------
# One training run (trial)
# ------------------------------
def train_once(hp, trial_seed):
    """
    hp keys:
      - name, hidden, act, dropout, batchnorm, lr, batch_size, epochs
    """
    set_seed(trial_seed)

    model = FFNN(G, I, hidden=hp["hidden"], act=hp["act"],
                 dropout=hp["dropout"], batchnorm=hp["batchnorm"]).to(DEVICE)
    model.apply(lambda m: init_linear(m, hp["act"]))

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    best_val = math.inf
    best_state = None
    noimp = 0
    train_curve, val_curve, eval_epochs = [], [], []

    t0 = time.time()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in batch_iter(tr_idx_t, batch_size=hp["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                pb = model(xb)
                loss = criterion(pb, yb)

            if USE_CUDA_AMP:
                SCALER.scale(loss).backward()
                if GRAD_CLIP is not None:
                    SCALER.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                SCALER.step(opt); SCALER.update()
            else:
                loss.backward()
                if GRAD_CLIP is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()

            total += float(loss.item()) * yb.size(0)
            seen  += yb.size(0)

        epoch_train = total / max(1, seen)
        train_curve.append(epoch_train)

        do_eval  = (epoch == 1) or (epoch % EVAL_EVERY == 0)
        do_print = (epoch == 1) or (epoch % LOG_EVERY  == 0)

        if do_eval:
            val_mse = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
            sch.step(val_mse)
            val_curve.append(val_mse)
            eval_epochs.append(epoch)

            if val_mse < best_val - 0.0:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1

        if do_print:
            if do_eval:
                print(f"[{hp['name']}] ep {epoch:03d} | act={hp['act']} | train {epoch_train:.5f} | val {val_mse:.5f} | lr {opt.param_groups[0]['lr']:.4g}")
            else:
                print(f"[{hp['name']}] ep {epoch:03d} | act={hp['act']} | train {epoch_train:.5f} | val - | lr {opt.param_groups[0]['lr']:.4g}")

        if do_eval and noimp >= PATIENCE:
            print(f"[{hp['name']}] Early stopping.")
            break

    if best_state: model.load_state_dict(best_state)

    # Final val metric (post-restore)
    final_val = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
    train_time = time.time() - t0

    # Per-trial plot
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{hp['name']}.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(range(1, len(train_curve)+1), train_curve, label="train MSE")
        if val_curve:
            plt.plot(eval_epochs, val_curve, "o-", label="val MSE")
        plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.title(f"{hp['name']} — act={hp['act']}")
        plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()
    except Exception as e:
        print(f"[WARN] Plot failed for {hp['name']}: {e}")

    rec = {
        "name": hp["name"],
        "hidden": hp["hidden"],
        "act": hp["act"],
        "dropout": hp["dropout"],
        "batchnorm": hp["batchnorm"],
        "lr": hp["lr"],
        "momentum": "-", "step_size": "-", "gamma": "-",
        "batch_size": hp["batch_size"],
        "epochs_trained": len(train_curve),
        "val_mse": float(final_val),
        "val_pearson": float("nan"),
        "train_time_sec": round(train_time, 1),
    }
    curves = {"train": train_curve, "val": val_curve, "val_epochs": eval_epochs}
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

def sample_hp(t):
    rnd = random.Random(SEED + 1000 + t)
    name, hidden = rnd.choice(ARCHS)
    act = ACTS[t % len(ACTS)]  # cycle activations deterministically
    return {
        "name": f"{name}_t{t}",
        "hidden": hidden,
        "act": act,
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
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
        "name","hidden","act","dropout","batchnorm","lr","momentum","step_size","gamma","batch_size",
        "epochs_trained","val_mse","val_pearson","train_time_sec"
    ])
    w.writeheader()

for t in range(N_TRIALS):
    hp = sample_hp(t)
    print(f"\n=== Trial {t+1}/{N_TRIALS}: {hp} ===")
    rec, model, curves = train_once(hp, trial_seed=SEED + t)

    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","hidden","act","dropout","batchnorm","lr","momentum","step_size","gamma","batch_size",
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
# Test only once (best model) + Pearson
# ------------------------------
print("\n[INFO] Evaluating test set for the best-by-val model only...")
best_bs = best_hp["batch_size"]
best_model.eval()
preds = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=best_bs, shuffle=False):
        with autocast_ctx():
            pb = best_model(xb)
        preds.append(pb.detach().cpu().numpy())
Y_test_np = Y[te_idx]
Y_pred_np = np.concatenate(preds, axis=0)
test_mse = float(((Y_test_np - Y_pred_np)**2).mean())
test_r   = pearson_mean_np(Y_test_np, Y_pred_np)
best_rec["test_mse"] = float(test_mse)
best_rec["test_pearson"] = float(test_r)
print(f"[BEST on TEST] act={best_hp['act']} | MSE: {test_mse:.6f} | r: {test_r:.4f}")

# Save a best-model curves plot including the test horizontal line
try:
    import matplotlib.pyplot as plt
    c = curves_by_name[best_rec["name"]]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(range(1, len(c["train"]) + 1), c["train"], label="train MSE")
    if c["val"]:
        plt.plot(c["val_epochs"], c["val"], label="val MSE")
    plt.axhline(best_rec["test_mse"], linestyle="--", linewidth=1.2, label=f"test MSE = {best_rec['test_mse']:.4g}")
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title(f"Best model — {best_rec['name']} (act={best_hp['act']})")
    plt.legend(); plt.tight_layout(); plt.savefig(BEST_FIG_CURVES, dpi=150); plt.close()
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
    plt.figure(figsize=(max(8, 0.42*len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE by model (activation swept)")
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
        plt.plot(range(1, len(c["train"])+1), c["train"], alpha=0.4, linewidth=1.0)
        if c["val"]:
            plt.plot(c["val_epochs"], c["val"], label=f"{r['name']}", linewidth=2.0)
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Top-5 models — validation curves")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot top-5 overlay: {e}")

print("[INFO] All done.")
