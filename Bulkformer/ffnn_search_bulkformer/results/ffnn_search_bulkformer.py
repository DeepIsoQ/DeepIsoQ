#!/usr/bin/env python3
"""
FFNN Random Search on BulkFormer Embeddings (GPU + AMP) — 2-stage

Stage 1:
- Random search over architectures (1–3 layers, widths in {256,512,1024,2048})
- Random activations, dropout, batchnorm, lr, batch_size
- Max 50 epochs, early stopping + global pruning from epoch >= 30
- 100 trials
- Saves:
    arch_search_results_bulkformer_stage1.csv
    arch_search_summary_bulkformer_stage1.json
    best_model_bulkformer_stage1.pt
    figs_trials_bulkformer/curves_stage1_*.png
    summary_val_mse_bar_bulkformer_stage1.png
    top5_val_curves_bulkformer_stage1.png

Stage 2:
- Loads top-5 configs from Stage 1 (in-memory)
- Retrains each for up to 200 epochs with early stopping (no global pruning)
- Saves:
    arch_search_results_bulkformer_stage2.csv
    arch_search_summary_bulkformer_stage2.json
    best_model_bulkformer_stage2.pt
    figs_trials_bulkformer/curves_stage2_*.png
    summary_val_mse_bar_bulkformer_stage2.png
    top5_val_curves_bulkformer_stage2.png

Both stages:
- Input X = X_bulkformer_emb (N, D_embed) from bulkformer_result.pt
- Targets Y = log1p(Y_tx)
- Compute test MSE + Pearson for best-by-val model in each stage
"""

import os, json, math, random, csv, time, pathlib
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

GRAD_CLIP  = 1.0
EVAL_EVERY = 5          # evaluate every N epochs

# Stage 1 config
STAGE1_N_TRIALS        = 100
STAGE1_MAX_EPOCHS      = 50
STAGE1_PATIENCE        = 5
STAGE1_MIN_PRUNE_EPOCH = 30
STAGE1_PRUNE_FACTOR    = 1.5

# Stage 2 config
STAGE2_TOP_K        = 5
STAGE2_MAX_EPOCHS   = 200
STAGE2_PATIENCE     = 10

# HPO search ranges (Stage 1)
BATCHES    = [128, 192, 256]
LRS        = [1e-3, 2e-3, 3e-3]
DROPOUTS   = [0.0, 0.1, 0.2]
BATCHNORMS = [False, True]
ACTS       = ["tanh", "relu", "gelu", "leakyrelu"]

# Architecture space: up to 3 layers, widths in this set
DEPTH_CHOICES = [1, 2, 3]
WIDTH_CHOICES = [256, 512, 1024, 2048]

# ------------------------------
# Output paths
# ------------------------------
BASE_DIR = "ffnn_search_bulkformer/results"
pathlib.Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
TRIAL_FIG_DIR   = os.path.join(BASE_DIR, "figs_trials_bulkformer")
pathlib.Path(TRIAL_FIG_DIR).mkdir(parents=True, exist_ok=True)

# Stage 1 paths
RESULTS_CSV_1      = os.path.join(BASE_DIR, "arch_search_results_bulkformer_stage1.csv")
SUMMARY_JSON_1     = os.path.join(BASE_DIR, "arch_search_summary_bulkformer_stage1.json")
SUMMARY_FIG_BAR_1  = os.path.join(BASE_DIR, "summary_val_mse_bar_bulkformer_stage1.png")
SUMMARY_FIG_TOP5_1 = os.path.join(BASE_DIR, "top5_val_curves_bulkformer_stage1.png")
BEST_MODEL_PT_1    = os.path.join(BASE_DIR, "best_model_bulkformer_stage1.pt")

# Stage 2 paths
RESULTS_CSV_2      = os.path.join(BASE_DIR, "arch_search_results_bulkformer_stage2.csv")
SUMMARY_JSON_2     = os.path.join(BASE_DIR, "arch_search_summary_bulkformer_stage2.json")
SUMMARY_FIG_BAR_2  = os.path.join(BASE_DIR, "summary_val_mse_bar_bulkformer_stage2.png")
SUMMARY_FIG_TOP5_2 = os.path.join(BASE_DIR, "top5_val_curves_bulkformer_stage2.png")
BEST_MODEL_PT_2    = os.path.join(BASE_DIR, "best_model_bulkformer_stage2.pt")

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
# Data path (BulkFormer result)
# ------------------------------
BF_PT = os.environ.get("BULKFORMER_PT")
if BF_PT is None:
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "BULKFORMER_PT not set and BLACKHOLE/USER env vars missing. "
            "Either export BULKFORMER_PT or run on DTU HPC where BLACKHOLE & USER are defined."
        )
    BF_PT = os.path.join(bh, user, "bulkformer_result.pt")

print(f"[INFO] Loading BulkFormer embeddings from: {BF_PT}", flush=True)
data = torch.load(BF_PT, map_location="cpu", weights_only=False)

# X: BulkFormer transcriptome-level embedding
X = data["X_bulkformer_emb"].float().cpu().numpy()    # (N, D_embed)
# Y: log1p of isoform expression
Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()  # (N, I)

N, D_embed = X.shape
_, I = Y.shape
print(f"[INFO] Shapes: X_bulkformer_emb={X.shape}, Y_tx(log1p)={Y.shape}", flush=True)
print(f"[INFO] N={N}, D_embed={D_embed}, I={I}", flush=True)

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
print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}", flush=True)

# ------------------------------
# Normalize once (train stats) + CPU tensors
# ------------------------------
X_mean = X[tr_idx].mean(axis=0)
X_std  = X[tr_idx].std(axis=0) + 1e-8
Xz = (X - X_mean) / X_std

Xt = torch.from_numpy(Xz).float()   # CPU
Yt = torch.from_numpy(Y ).float()   # CPU

tr_idx_t = torch.from_numpy(tr_idx)  # CPU
va_idx_t = torch.from_numpy(va_idx)  # CPU
te_idx_t = torch.from_numpy(te_idx)  # CPU

def batch_iter(idxs_t, batch_size, shuffle=True):
    idxs = idxs_t
    if shuffle:
        perm = torch.randperm(idxs.numel())
        idxs = idxs[perm]

    for i in range(0, idxs.numel(), batch_size):
        j = idxs[i:i+batch_size]
        xb = Xt.index_select(0, j).to(DEVICE, non_blocking=True)
        yb = Yt.index_select(0, j).to(DEVICE, non_blocking=True)
        yield xb, yb

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
            with torch.amp.autocast("cuda", enabled=AMP):
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
# Architecture sampling (Stage 1)
# ------------------------------
def sample_arch(rnd: random.Random):
    depth = rnd.choice(DEPTH_CHOICES)
    hidden = [rnd.choice(WIDTH_CHOICES) for _ in range(depth)]
    return hidden

def sample_hp_stage1(t):
    rnd = random.Random(SEED + 2000 + t)
    hidden = sample_arch(rnd)
    arch_name = f"L{len(hidden)}_" + "_".join(map(str, hidden))
    return {
        "name": f"{arch_name}_BF_t{t}",
        "hidden": hidden,
        "act": rnd.choice(ACTS),
        "dropout": rnd.choice(DROPOUTS),
        "batchnorm": rnd.choice(BATCHNORMS),
        "lr": rnd.choice(LRS),
        "batch_size": rnd.choice(BATCHES),
        "epochs": STAGE1_MAX_EPOCHS,
    }

# ------------------------------
# One training run (per trial)
# ------------------------------
def train_once(hp, trial_seed, global_best_val=None, stage=1):
    """
    hp keys:
      - name, hidden, act, dropout, batchnorm, lr, batch_size, epochs
    stage: 1 (with global pruning) or 2 (no global pruning)
    """
    set_seed(trial_seed)

    model = FFNN(
        D_embed, I,
        hidden=hp["hidden"],
        act=hp["act"],
        dropout=hp["dropout"],
        batchnorm=hp["batchnorm"]
    ).to(DEVICE)
    model.apply(lambda m: init_linear(m, hp["act"]))

    opt    = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sch    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    if stage == 1:
        max_epochs = STAGE1_MAX_EPOCHS
        patience   = STAGE1_PATIENCE
    else:
        max_epochs = STAGE2_MAX_EPOCHS
        patience   = STAGE2_PATIENCE

    best_val   = math.inf
    best_state = None
    noimp      = 0
    train_curve, val_curve = [], []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for xb, yb in batch_iter(tr_idx_t, batch_size=hp["batch_size"], shuffle=True):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
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

            # Per-model best tracking
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
                epoch >= STAGE1_MIN_PRUNE_EPOCH and
                val_mse > STAGE1_PRUNE_FACTOR * global_best_val
            ):
                print(
                    f"[{hp['name']}] Pruned at epoch {epoch} "
                    f"(val {val_mse:.5f} >> best {global_best_val:.5f}).",
                    flush=True,
                )
                break

            print(
                f"[{hp['name']}] ep {epoch:03d} | act={hp['act']} | "
                f"train {epoch_train:.5f} | val {val_mse:.5f} | "
                f"lr {opt.param_groups[0]['lr']:.4g}",
                flush=True,
            )
            if noimp >= patience:
                print(f"[{hp['name']}] Early stopping (no improvement).", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse    = evaluate_on(model, va_idx_t, batch_size=hp["batch_size"])
    train_time = time.time() - t0

    # Per-trial plot
    try:
        import matplotlib.pyplot as plt
        stage_tag = "stage1" if stage == 1 else "stage2"
        fig_path = os.path.join(TRIAL_FIG_DIR, f"curves_{stage_tag}_{hp['name']}.png")
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
        plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.title(f"{stage_tag}: {hp['name']} — act={hp['act']}")
        plt.legend(); plt.tight_layout()
        plt.savefig(fig_path, dpi=150); plt.close()
    except Exception as e:
        print(f"[WARN] Plot failed for {hp['name']}: {e}", flush=True)

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
        "val_pearson": float("nan"),
        "train_time_sec": round(train_time, 1),
    }
    curves = {"train": train_curve, "val": val_curve}
    return rec, model, curves, best_val

# ------------------------------
# Stage 1: random search
# ------------------------------
print(f"[INFO] Stage 1: random search on BulkFormer embeddings with {STAGE1_N_TRIALS} trials...", flush=True)
results1, curves_by_name_1 = [], {}
best_rec_1, best_model_1, best_hp_1 = None, None, None
GLOBAL_BEST_VAL_1 = math.inf

# CSV header (stage 1)
with open(RESULTS_CSV_1, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name","hidden","act","dropout","batchnorm","lr",
        "momentum","step_size","gamma","batch_size",
        "epochs_trained","val_mse","val_pearson","train_time_sec"
    ])
    w.writeheader()

for t in range(STAGE1_N_TRIALS):
    hp = sample_hp_stage1(t)
    print(f"\n=== Stage 1 — Trial {t+1}/{STAGE1_N_TRIALS}: {hp} ===", flush=True)
    rec, model, curves, trial_best_val = train_once(
        hp,
        trial_seed=SEED + t,
        global_best_val=GLOBAL_BEST_VAL_1,
        stage=1,
    )

    with open(RESULTS_CSV_1, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","hidden","act","dropout","batchnorm","lr",
            "momentum","step_size","gamma","batch_size",
            "epochs_trained","val_mse","val_pearson","train_time_sec"
        ])
        w.writerow(rec)

    results1.append(rec)
    curves_by_name_1[rec["name"]] = curves

    GLOBAL_BEST_VAL_1 = min(GLOBAL_BEST_VAL_1, trial_best_val)

    if best_rec_1 is None or rec["val_mse"] < best_rec_1["val_mse"]:
        best_rec_1   = rec
        best_model_1 = model
        best_hp_1    = hp
        torch.save({
            "state_dict": best_model_1.state_dict(),
            "hparams": {k: v for k, v in hp.items()},
            "X_mean": X_mean, "X_std": X_std,
            "meta": {"D_embed": D_embed, "I": I, "seed": SEED, "stage": 1}
        }, BEST_MODEL_PT_1)
        print(f"[BEST][Stage 1] Updated best by Val: {rec['name']} (val_mse={rec['val_mse']:.6f})", flush=True)

# ------------------------------
# Stage 1: test evaluation for best model
# ------------------------------
print("\n[INFO] Stage 1: evaluating test set for the best-by-val model...", flush=True)
best_bs_1 = best_hp_1["batch_size"]
best_model_1.eval()

preds_1 = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=best_bs_1, shuffle=False):
        with torch.amp.autocast("cuda", enabled=AMP):
            preds_1.append(best_model_1(xb))

Y_test_1 = Yt[te_idx_t].to(DEVICE)
Y_pred_1 = torch.cat(preds_1, dim=0)

test_mse_1 = float(((Y_test_1 - Y_pred_1)**2).mean().item())
test_r_1   = pearson_mean_gpu(Y_test_1, Y_pred_1)

best_rec_1["test_mse"]     = test_mse_1
best_rec_1["test_pearson"] = test_r_1

print(f"[BEST on TEST][Stage 1] BulkFormer FFNN | act={best_hp_1['act']} | "
      f"MSE: {test_mse_1:.6f} | r: {test_r_1:.4f}", flush=True)

# ------------------------------
# Stage 1: summary + plots
# ------------------------------
results1_sorted = sorted(results1, key=lambda r: r["val_mse"])
summary1 = {
    "stage": 1,
    "best": best_rec_1,
    "top5": results1_sorted[:5],
    "n_trials": STAGE1_N_TRIALS,
    "csv": RESULTS_CSV_1,
    "best_model_pt": BEST_MODEL_PT_1,
}
with open(SUMMARY_JSON_1, "w") as f:
    json.dump(summary1, f, indent=2)

print("\n=== STAGE 1 COMPLETE (BulkFormer FFNN) ===", flush=True)
print(json.dumps(summary1, indent=2), flush=True)

# Stage 1 plots
try:
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results1_sorted]
    vals  = [r["val_mse"] for r in results1_sorted]
    plt.figure(figsize=(max(8, 0.4*len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE by model (BulkFormer embeddings — Stage 1)")
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_BAR_1, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot Stage 1 summary bar: {e}", flush=True)

try:
    import matplotlib.pyplot as plt
    top5 = results1_sorted[:5]
    plt.figure(figsize=(8, 5))
    for r in top5:
        c = curves_by_name_1[r["name"]]
        xs_v = [
            e for e in range(1, len(c["train"])+1)
            if e == 1 or e % EVAL_EVERY == 0
        ][:len(c["val"])]
        plt.plot(xs_v, c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        plt.plot(range(1, len(c["train"])+1), c["train"], alpha=0.4, linewidth=1.0)
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Top-5 models — validation curves (BulkFormer embeddings — Stage 1)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5_1, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot Stage 1 top-5 overlay: {e}", flush=True)

# ------------------------------
# Stage 2: retrain top-K from Stage 1
# ------------------------------
print(f"\n[INFO] Stage 2: retraining top-{STAGE2_TOP_K} configs from Stage 1...", flush=True)

top_stage1 = results1_sorted[:STAGE2_TOP_K]
top_hps_2 = []
for r in top_stage1:
    hp2 = {
        "name": f"{r['name']}_stage2",
        "hidden": r["hidden"],          # list, still in-memory
        "act": r["act"],
        "dropout": float(r["dropout"]),
        "batchnorm": bool(r["batchnorm"]),
        "lr": float(r["lr"]),
        "batch_size": int(r["batch_size"]),
        "epochs": STAGE2_MAX_EPOCHS,
    }
    top_hps_2.append(hp2)

results2, curves_by_name_2 = [], {}
best_rec_2, best_model_2, best_hp_2 = None, None, None

# CSV header (stage 2)
with open(RESULTS_CSV_2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "name","hidden","act","dropout","batchnorm","lr",
        "momentum","step_size","gamma","batch_size",
        "epochs_trained","val_mse","val_pearson","train_time_sec"
    ])
    w.writeheader()

for t, hp in enumerate(top_hps_2):
    print(f"\n=== Stage 2 — Trial {t+1}/{len(top_hps_2)}: {hp} ===", flush=True)
    rec, model, curves, _ = train_once(
        hp,
        trial_seed=SEED + 1000 + t,
        global_best_val=None,
        stage=2,
    )

    with open(RESULTS_CSV_2, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name","hidden","act","dropout","batchnorm","lr",
            "momentum","step_size","gamma","batch_size",
            "epochs_trained","val_mse","val_pearson","train_time_sec"
        ])
        w.writerow(rec)

    results2.append(rec)
    curves_by_name_2[rec["name"]] = curves

    if best_rec_2 is None or rec["val_mse"] < best_rec_2["val_mse"]:
        best_rec_2   = rec
        best_model_2 = model
        best_hp_2    = hp
        torch.save({
            "state_dict": best_model_2.state_dict(),
            "hparams": {k: v for k, v in hp.items()},
            "X_mean": X_mean, "X_std": X_std,
            "meta": {"D_embed": D_embed, "I": I, "seed": SEED, "stage": 2}
        }, BEST_MODEL_PT_2)
        print(f"[BEST][Stage 2] Updated best by Val: {rec['name']} (val_mse={rec['val_mse']:.6f})", flush=True)

# ------------------------------
# Stage 2: test evaluation for best model
# ------------------------------
print("\n[INFO] Stage 2: evaluating test set for the best-by-val model...", flush=True)
best_bs_2 = best_hp_2["batch_size"]
best_model_2.eval()

preds_2 = []
with torch.no_grad():
    for xb, _ in batch_iter(te_idx_t, batch_size=best_bs_2, shuffle=False):
        with torch.amp.autocast("cuda", enabled=AMP):
            preds_2.append(best_model_2(xb))

Y_test_2 = Yt[te_idx_t].to(DEVICE)
Y_pred_2 = torch.cat(preds_2, dim=0)

test_mse_2 = float(((Y_test_2 - Y_pred_2)**2).mean().item())
test_r_2   = pearson_mean_gpu(Y_test_2, Y_pred_2)

best_rec_2["test_mse"]     = test_mse_2
best_rec_2["test_pearson"] = test_r_2

print(f"[BEST on TEST][Stage 2] BulkFormer FFNN | act={best_hp_2['act']} | "
      f"MSE: {test_mse_2:.6f} | r: {test_r_2:.4f}", flush=True)

# ------------------------------
# Stage 2: summary + plots
# ------------------------------
results2_sorted = sorted(results2, key=lambda r: r["val_mse"])
summary2 = {
    "stage": 2,
    "best": best_rec_2,
    "top5": results2_sorted[:5],
    "n_trials": len(top_hps_2),
    "csv": RESULTS_CSV_2,
    "best_model_pt": BEST_MODEL_PT_2,
}
with open(SUMMARY_JSON_2, "w") as f:
    json.dump(summary2, f, indent=2)

print("\n=== STAGE 2 COMPLETE (BulkFormer FFNN) ===", flush=True)
print(json.dumps(summary2, indent=2), flush=True)

# Stage 2 plots
try:
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results2_sorted]
    vals  = [r["val_mse"] for r in results2_sorted]
    plt.figure(figsize=(max(8, 0.4*len(names)), 4.8))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE by model (BulkFormer embeddings — Stage 2)")
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_BAR_2, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot Stage 2 summary bar: {e}", flush=True)

try:
    import matplotlib.pyplot as plt
    top5 = results2_sorted[:5]
    plt.figure(figsize=(8, 5))
    for r in top5:
        c = curves_by_name_2[r["name"]]
        xs_v = [
            e for e in range(1, len(c["train"])+1)
            if e == 1 or e % EVAL_EVERY == 0
        ][:len(c["val"])]
        plt.plot(xs_v, c["val"], label=f"{r['name']} (val)", linewidth=2.0)
        plt.plot(range(1, len(c["train"])+1), c["train"], alpha=0.4, linewidth=1.0)
    plt.xlabel("epoch"); plt.ylabel("MSE")
    plt.title("Top-5 models — validation curves (BulkFormer embeddings — Stage 2)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SUMMARY_FIG_TOP5_2, dpi=150)
    plt.close()
except Exception as e:
    print(f"[WARN] Could not plot Stage 2 top-5 overlay: {e}", flush=True)

print("[INFO] All done (BulkFormer 2-stage FFNN search).", flush=True)
