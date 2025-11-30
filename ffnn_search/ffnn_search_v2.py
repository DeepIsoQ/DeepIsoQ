#!/usr/bin/env python3
"""
FFNN Random Search (GPU + AMP) — 2-stage

Stage 1:
- Random search over architectures (1–3 layers, widths in {256,512,1024,2048})
- Random activations, dropout, batchnorm, lr, batch_size
- Max 50 epochs, early stopping + global pruning from epoch >= 30
- 100 trials
- Saves: arch_search_results_stage1.csv, arch_search_summary_stage1.json, best_model_stage1.pt

Stage 2:
- Loads top-K configs from Stage-1 CSV (by val_mse)
- Retrains each for up to 200 epochs with early stopping (no global pruning)
- K = 5
- Saves: arch_search_results_stage2.csv, arch_search_summary_stage2.json, best_model_stage2.pt

Both stages:
- Use log1p(Y_tx) as targets, standardized Xg_log1p as inputs
- Compute test MSE + Pearson for best-by-val model in that stage
"""

import os, json, math, random, csv, time, pathlib, ast
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# ------------------------------
# Global config
# ------------------------------
SEED      = 42
TEST_FRAC = 0.15
VAL_FRAC  = 0.15
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP       = (DEVICE == "cuda")

# ---- STAGE CONTROL ----
# STAGE = 1 → random search with pruning
# STAGE = 2 → retrain top-K configs from Stage 1 (no global pruning)
STAGE        = 2          # set to 1 or 2
TOP_K_STAGE2 = 5          # number of best configs from Stage 1 to retrain in Stage 2

GRAD_CLIP    = 1.0
EVAL_EVERY   = 5          # evaluate every N epochs (both stages)

# HPO search ranges (Stage 1)
BATCHES    = [128, 192, 256]
LRS        = [1e-3, 2e-3, 3e-3]
DROPOUTS   = [0.0, 0.1, 0.2]
BATCHNORMS = [False, True]
ACTS       = ["tanh", "relu", "gelu", "leakyrelu"]

# Architecture space: up to 3 layers, widths in this set
DEPTH_CHOICES = [1, 2, 3]
WIDTH_CHOICES = [256, 512, 1024, 2048]

# Stage-specific config
if STAGE == 1:
    N_TRIALS        = 100
    MAX_EPOCHS      = 50
    PATIENCE        = 5
    MIN_PRUNE_EPOCH = 30   # don't prune before this epoch
    PRUNE_FACTOR    = 1.5  # prune if val is > 1.5x global best
elif STAGE == 2:
    # N_TRIALS will be set after loading top-K from Stage-1 CSV
    N_TRIALS        = TOP_K_STAGE2
    MAX_EPOCHS      = 200
    PATIENCE        = 10
    MIN_PRUNE_EPOCH = None
    PRUNE_FACTOR    = None
else:
    raise ValueError("STAGE must be 1 or 2")

# ------------------------------
# Output paths (stage-aware)
# ------------------------------
BASE_DIR = "ffnn_search/results"
os.makedirs(BASE_DIR, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
TRIAL_FIG_DIR   = os.path.join(BASE_DIR, "figs_trials")
BEST_FIG_CURVES = os.path.join(BASE_DIR, "best_model_curves.png")

if STAGE == 1:
    RESULTS_CSV      = os.path.join(BASE_DIR, "arch_search_results_stage1.csv")
    SUMMARY_JSON     = os.path.join(BASE_DIR, "arch_search_summary_stage1.json")
    SUMMARY_FIG_BAR  = os.path.join(BASE_DIR, "summary_val_mse_bar_stage1.png")
    SUMMARY_FIG_TOP5 = os.path.join(BASE_DIR, "top5_val_curves_stage1.png")
    BEST_MODEL_PT    = os.path.join(BASE_DIR, "best_model_stage1.pt")
elif STAGE == 2:
    RESULTS_CSV      = os.path.join(BASE_DIR, "arch_search_results_stage2.csv")
    SUMMARY_JSON     = os.path.join(BASE_DIR, "arch_search_summary_stage2.json")
    SUMMARY_FIG_BAR  = os.path.join(BASE_DIR, "summary_val_mse_bar_stage2.png")
    SUMMARY_FIG_TOP5 = os.path.join(BASE_DIR, "top5_val_curves_stage2.png")
    BEST_MODEL_PT    = os.path.join(BASE_DIR, "best_model_stage2.pt")

STAGE1_RESULTS_CSV = os.path.join(BASE_DIR, "arch_search_results_stage1.csv")

# ------------------------------
# Repro & matmul knobs
# ------------------------------
def set_seed(s):
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

X = data["Xg_log1p"].float().cpu().numpy()
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()

N, G = X.shape
_, I = Y.shape
print(f"[INFO] Shapes: X={X.shape}, Y={Y.shape}")

# ------------------------------
# Split: train / val / test
# ------------------------------
all_idx = np.arange(N)
trval_idx, te_idx = train_test_split(
    all_idx, test_size=TEST_FRAC, random_state=SEED, shuffle=True
)
val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
tr_idx, va_idx = train_test_split(
    trval_idx, test_size=val_rel, random_state=SEED, shuffle=True
)
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

# ------------------------------
# Normalize once (train stats) + device tensors
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
# Activations + init
# ------------------------------
def get_activation(name: str):
    n = name.lower()
    if n == "relu":        return nn.ReLU()
    if n == "gelu":        return nn.GELU()
    if n in ("leakyrelu", "leaky_relu"): return nn.LeakyReLU(0.01)
    if n == "tanh":        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def init_linear(m, act: str):
    if not isinstance(m, nn.Linear):
        return
    a = act.lower()
    if a == "tanh":
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
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.act_name = act

    def forward(self, x):
        return self.net(x)

# ------------------------------
# Metrics / Eval
# ------------------------------
criterion = nn.MSELoss()

def evaluate_on(model, idxs_t, batch_size):
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
    yt = Y_true - Y_true.mean(dim=0, keepdim=True)
    yp = Y_pred - Y_pred.mean(dim=0, keepdim=True)
    num = (yt * yp).sum(dim=0)
    den = torch.sqrt((yt * yt).sum(dim=0)) * torch.sqrt((yp * yp).sum(dim=0)) + 1e-8
    r = num / den
    return r.nanmean().item()

# ------------------------------
# Architecture sampling (Stage 1)
# ------------------------------
def sample_arch(rnd: random.Random):
    depth = rnd.choice(DEPTH_CHOICES)
    hidden = [rnd.choice(WIDTH_CHOICES) for _ in range(depth)]
    return hidden

def sample_hp(t):
    rnd = random.Random(SEED + 1000 + t)
    hidden = sample_arch(rnd)
    arch_name = f"L{len(hidden)}_" + "_".join(map(str, hidden))
    return {
        "name": f"{arch_name}_t{t}",
        "hidden": hidden,
        "act": rnd.choice(ACTS),
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
        "batch_size": rnd.choice(BATCHES),
        "epochs": MAX_EPOCHS,
    }

# ------------------------------
# One training run (per trial)
# ------------------------------
def train_once(hp, trial_seed, global_best_val=None, stage=1):
    set_seed(trial_seed)

    model = FFNN(
        G, I,
        hidden=hp["hidden"],
        act=hp["act"],
        dropout=hp["dropout"],
        batchnorm=hp["batchnorm"]
    ).to(DEVICE)
    model.apply(lambda m: init_linear(m, hp["act"]))

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

        epoch_train = total / max(1, seen)
        train_curve.append(epoch_train)

        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_mse = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
            val_curve.append(val_mse)
            sch.step(val_mse)

            if val_mse < best_val - 0.0:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1

            # Global pruning (Stage 1 only)
            if (
                stage == 1 and
                global_best_val is not None and
                global_best_val < math.inf and
                (MIN_PRUNE_EPOCH is not None) and
                epoch >= MIN_PRUNE_EPOCH and
                val_mse > PRUNE_FACTOR * global_best_val
            ):
                print(
                    f"[{hp['name']}] Pruned at epoch {epoch} "
                    f"(val {val_mse:.5f} >> best {global_best_val:.5f})."
                )
                break

            print(
                f"[{hp['name']}] ep {epoch:03d} | act={hp['act']} | "
                f"train {epoch_train:.5f} | val {val_mse:.5f} | "
                f"lr {opt.param_groups[0]['lr']:.4g}"
            )
            if noimp >= PATIENCE:
                print(f"[{hp['name']}] Early stopping (no improvement).")
                break

    # Reload best state (if any) and compute final val MSE + val Pearson
    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse    = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
    train_time = time.time() - t0

    # --- NEW: compute validation Pearson ---
    model.eval()
    with torch.no_grad():
        preds_val = []
        for xb, _ in batch_iter(va_idx_t, batch_size=hp["batch_size"], shuffle=False):
            with torch.cuda.amp.autocast(enabled=AMP):
                preds_val.append(model(xb))
        Y_val_t    = Yt[va_idx_t]
        Y_val_pred = torch.cat(preds_val, dim=0)
        val_r      = pearson_mean_gpu(Y_val_t, Y_val_pred)

    # Per-trial plot
    pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{hp['name']}.png")
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(train_curve, label="train MSE")
        xs_v = [
            e for e in range(1, len(train_curve)+1)
            if e == 1 or e % EVAL_EVERY == 0
        ][:len(val_curve)]
        if len(xs_v) == len(val_curve):
            plt.plot(xs_v, val_curve, "o-", label="val MSE")
        else:
            plt.plot(val_curve, "o-", label="val MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title(f"{hp['name']} — act={hp['act']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
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
        "val_mse": float(val_mse),
        "val_pearson": float(val_r),   
        "train_time_sec": round(train_time, 1),
    }
    curves = {"train": train_curve, "val": val_curve}
    return rec, model, curves, best_val

# ------------------------------
# Stage 2: load top-K hparams from Stage 1
# ------------------------------
top_hps = None
if STAGE == 2:
    if not os.path.exists(STAGE1_RESULTS_CSV):
        raise FileNotFoundError(
            f"Stage 1 results CSV not found at {STAGE1_RESULTS_CSV}. "
            "Run Stage 1 first."
        )
    with open(STAGE1_RESULTS_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise RuntimeError("Stage 1 results CSV is empty.")

    rows_sorted = sorted(rows, key=lambda r: float(r["val_mse"]))
    rows_top    = rows_sorted[:TOP_K_STAGE2]

    top_hps = []
    for r in rows_top:
        hidden = ast.literal_eval(r["hidden"])
        hp = {
            "name": f"{r['name']}_stage2",
            "hidden": hidden,
            "act": r["act"],
            "dropout": float(r["dropout"]),
            "batchnorm": (str(r["batchnorm"]) in ["True", "true", "1"]),
            "lr": float(r["lr"]),
            "batch_size": int(r["batch_size"]),
            "epochs": MAX_EPOCHS,
        }
        top_hps.append(hp)

    N_TRIALS = len(top_hps)
    print(f"[INFO] Stage 2: loaded top-{N_TRIALS} configs from Stage 1.")

# ------------------------------
# Run search / retrain
# ------------------------------
if STAGE == 1:
    print(f"[INFO] Stage 1: random search with {N_TRIALS} trials...")
else:
    print(f"[INFO] Stage 2: retraining {N_TRIALS} best configs from Stage 1...")

results, curves_by_name = [], {}
best_rec, best_model, best_hp = None, None, None
GLOBAL_BEST_VAL = math.inf if STAGE == 1 else None

# CSV header
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name","hidden","act","dropout","batchnorm","lr",
        "momentum","step_size","gamma","batch_size",
        "epochs_trained","val_mse","val_pearson","train_time_sec"
    ])
    w.writeheader()

for t in range(N_TRIALS):
    if STAGE == 1:
        hp = sample_hp(t)
    else:
        hp = top_hps[t]

    print(f"\n=== Trial {t+1}/{N_TRIALS}: {hp} ===")
    rec, model, curves, trial_best_val = train_once(
        hp,
        trial_seed=SEED + t,
        global_best_val=GLOBAL_BEST_VAL,
        stage=STAGE,
    )

    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","hidden","act","dropout","batchnorm","lr",
            "momentum","step_size","gamma","batch_size",
            "epochs_trained","val_mse","val_pearson","train_time_sec"
        ])
        w.writerow(rec)

    results.append(rec)
    curves_by_name[rec["name"]] = curves

    if STAGE == 1:
        GLOBAL_BEST_VAL = min(GLOBAL_BEST_VAL, trial_best_val)

    if best_rec is None or rec["val_mse"] < best_rec["val_mse"]:
        best_rec   = rec
        best_model = model
        best_hp    = hp
        torch.save({
            "state_dict": best_model.state_dict(),
            "hparams": {k: v for k, v in hp.items()},

            "X_mean": X_mean,
            "X_std": X_std,
            "meta": {"G": G, "I": I, "seed": SEED, "stage": STAGE}
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

Y_test_t = Yt[te_idx_t]
Y_pred_t = torch.cat(preds, dim=0)

test_mse = float(((Y_test_t - Y_pred_t) ** 2).mean().item())
test_r   = pearson_mean_gpu(Y_test_t, Y_pred_t)

best_rec["test_mse"]     = test_mse
best_rec["test_pearson"] = test_r

print(f"[BEST on TEST] stage={STAGE} | act={best_hp['act']} | "
      f"MSE: {test_mse:.6f} | r: {test_r:.4f}")

# ------------------------------
# Summary files + summary plots
# ------------------------------
results_sorted = sorted(results, key=lambda r: r["val_mse"])
summary = {
    "stage": STAGE,
    "best": best_rec,
    "top5": results_sorted[:5],
    "n_trials": N_TRIALS,
    "csv": RESULTS_CSV,
    "best_model_pt": BEST_MODEL_PT,
}
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SEARCH / RETRAIN COMPLETE ===")
print(json.dumps(summary, indent=2))

# Bar chart of Val MSE (all models)
try:
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results_sorted]
    vals  = [r["val_mse"] for r in results_sorted]
    plt.figure(figsize=(max(8, 0.4 * len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation MSE by model (stage {STAGE})")
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
        xs_v = [
            e for e in range(1, len(c["train"]) + 1)
            if e == 1 or e % EVAL_EVERY == 0
        ][:len(c["val"])]
        plt.plot(xs_v, c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        plt.plot(range(1, len(c["train"]) + 1), c["train"], alpha=0.4, linewidth=1.0)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(f"Top-5 models — validation curves (stage {STAGE})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot top-5 overlay: {e}")

print("[INFO] All done.")
