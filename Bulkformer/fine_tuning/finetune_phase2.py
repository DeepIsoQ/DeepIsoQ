#!/usr/bin/env python3
"""
Phase 2: Safe fine-tuning of BulkFormer + FFNN head

- Loads original gene expression (Xg_log1p) and isoform targets (Y_tx)
- Aligns genes to BulkFormer gene list
- Rebuilds BulkFormer with pretrained checkpoint
- Rebuilds FFNN head from Phase-1 best model (on BulkFormer embeddings)
- Freezes all BulkFormer parameters EXCEPT the last GBFormer block
- Fine-tunes end-to-end with batch_size=1 to avoid OOM
- Uses the same embedding pooling as in the extractor (max+mean+median over HVGs)
- Uses the same embedding normalization (X_mean, X_std) as Phase-1

Saves:
- results_phase2_finetune.pt   → fine-tuned encoder + head
- finetune_summary.json        → MSE / Pearson on train/val/test
"""

import os, json, math, time, random, pathlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import OrderedDict
from torch_geometric.typing import SparseTensor
from isoform_model import IsoformFineTuneModel

# ------------------------------
# Config
# ------------------------------
SEED            = 42
TEST_FRAC       = 0.15
VAL_FRAC        = 0.15
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AMP             = True          
GRAD_CLIP       = 1.0

MAX_EPOCHS_FT   = 10
PATIENCE_FT     = 3
EVAL_EVERY      = 1
BATCH_SIZE_FT   = 1              # keep =1 to avoid OOM

LR_ENCODER      = 1e-5           # last GBFormer block
LR_HEAD         = 5e-4           # FFNN head

# ------------------------------
# Paths
# ------------------------------

# 1) Phase-1 best FFNN on BulkFormer embeddings
BEST_FFN_PT = "/zhome/02/d/213485/DeepIsoQ/Bulkformer/ffnn_search_bulkformer/results/best_model_bulkformer.pt"

# 2) BulkFormer repo with model/ and data/
BULKFORMER_ROOT = "/dtu/blackhole/0d/213485/s243310/BulkFormer"

GRAPH_PATH            = os.path.join(BULKFORMER_ROOT, "data",  "G_gtex.pt")
WEIGHTS_PATH          = os.path.join(BULKFORMER_ROOT, "data",  "G_gtex_weight.pt")
GENE_EMB_PATH         = os.path.join(BULKFORMER_ROOT, "data",  "esm2_feature_concat.pt")
CKPT_PATH             = os.path.join(BULKFORMER_ROOT, "model", "Bulkformer_ckpt_epoch_29.pt")
BULKFORMER_GENE_CSV   = os.path.join(BULKFORMER_ROOT, "data",  "bulkformer_gene_info.csv")
HIGH_VAR_GENE_IDX_PT  = os.path.join(BULKFORMER_ROOT, "data",  "high_var_gene_list.pt")

# 3) Original data.pt (gene expression + isoforms)
DATA_PT = os.environ.get("DATA_PT")
if DATA_PT is None:
    bh = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")
    if bh is None or user is None:
        raise RuntimeError(
            "DATA_PT not set and BLACKHOLE/USER env vars missing. "
            "Set DATA_PT to your /path/to/data.pt"
        )
    DATA_PT = os.path.join(bh, user, "data.pt")

# 4) Outputs
RESULTS_DIR = "results"
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
OUT_PT       = os.path.join(RESULTS_DIR, "results_phase2_finetune.pt")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "finetune_summary.json")

# ------------------------------
# Repro
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
# Imports from BulkFormer repo
# ------------------------------
from utils.BulkFormer import BulkFormer
from model.config import model_params

# ------------------------------
# Helper: align genes to BulkFormer gene list
# ------------------------------
def main_gene_selection(X_df: pd.DataFrame, gene_list):
    """
    Align expression matrix to BulkFormer gene list (gene_list).
    Missing genes are filled with a placeholder value -10.

    Returns:
      - X_df_full: samples x gene_list (in order)
      - to_fill_columns: list of genes that were missing and thus imputed
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(
        np.full((X_df.shape[0], len(to_fill_columns)), -10, dtype=np.float32),
        columns=to_fill_columns,
        index=X_df.index,
    )

    X_df_full = pd.DataFrame(
        np.concatenate([X_df.values, padding_df.values], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns),
    )

    X_df_full = X_df_full[gene_list]
    return X_df_full, to_fill_columns

# ------------------------------
# Activations + FFNN
# ------------------------------
def get_activation(name: str):
    n = name.lower()
    if n == "relu":        return nn.ReLU()
    if n == "gelu":        return nn.GELU()
    if n in ("leakyrelu", "leaky_relu"): return nn.LeakyReLU(0.01)
    if n == "tanh":        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

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
# Batch iterator
# ------------------------------
def batch_iter(Xt, Yt, idxs_t, batch_size, shuffle=True):
    if shuffle:
        idxs_t = idxs_t[torch.randperm(idxs_t.numel(), device=idxs_t.device)]
    for i in range(0, idxs_t.numel(), batch_size):
        j = idxs_t[i:i+batch_size]
        yield Xt.index_select(0, j), Yt.index_select(0, j)

# ------------------------------
# Metrics / Eval
# ------------------------------
criterion = nn.MSELoss()

def evaluate_on(model, Xt, Yt, idxs_t, batch_size):
    model.eval()
    total, seen = 0.0, 0
    with torch.no_grad():
        for xb, yb in batch_iter(Xt, Yt, idxs_t, batch_size=batch_size, shuffle=False):
            if AMP and DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    pb = model(xb)
                    loss = criterion(pb, yb)
            else:
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
# Main
# ------------------------------
if __name__ == "__main__":
    print(f"[INFO] DEVICE: {DEVICE}")
    print(f"[INFO] Loading data.pt from: {DATA_PT}")
    data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

    Xg = data["Xg_log1p"].float().cpu().numpy()
    Y  = torch.log1p(data["Y_tx"].float()).cpu().numpy()
    gene_ids = list(data["gene_ids"])
    N, G_orig = Xg.shape
    _, I = Y.shape
    print(f"[INFO] Shapes: Xg_log1p={Xg.shape}, Y(log1p)={Y.shape}")

    # --------------------------
    # Align genes to BulkFormer list
    # --------------------------
    print("[INFO] Loading BulkFormer gene list...")
    bulk_gene_info = pd.read_csv(BULKFORMER_GENE_CSV)
    bulk_gene_list = bulk_gene_info["ensg_id"].astype(str).tolist()

    print("[INFO] Aligning genes to BulkFormer gene list...")
    X_df = pd.DataFrame(Xg, columns=gene_ids,
                        index=[f"sample_{i}" for i in range(N)])
    X_aligned_df, to_fill = main_gene_selection(X_df, bulk_gene_list)
    X_aligned = X_aligned_df.values.astype(np.float32)
    print(f"[INFO] Aligned expression shape: {X_aligned.shape}")
    print(f"[INFO] Missing genes imputed with -10: {len(to_fill)}")

    # --------------------------
    # Train/Val/Test split
    # --------------------------
    all_idx = np.arange(N)
    trval_idx, te_idx = train_test_split(all_idx, test_size=TEST_FRAC,
                                         random_state=SEED, shuffle=True)
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    tr_idx, va_idx = train_test_split(trval_idx, test_size=val_rel,
                                      random_state=SEED, shuffle=True)
    print(f"[INFO] Split sizes: train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    # --------------------------
    # Tensors on DEVICE
    # --------------------------
    Xt_all = torch.from_numpy(X_aligned).to(DEVICE).float()
    Yt_all = torch.from_numpy(Y).to(DEVICE).float()

    tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
    va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
    te_idx_t = torch.from_numpy(te_idx).to(DEVICE)

    # --------------------------
    # Load BulkFormer
    # --------------------------
    print("[INFO] Initializing BulkFormer backbone...")
    graph_raw   = torch.load(GRAPH_PATH,   map_location="cpu")
    weights_raw = torch.load(WEIGHTS_PATH, map_location="cpu")
    graph = SparseTensor(row=graph_raw[1], col=graph_raw[0], value=weights_raw).t().to(DEVICE)

    gene_emb = torch.load(GENE_EMB_PATH, map_location="cpu")

    model_params["graph"]    = graph
    model_params["gene_emb"] = gene_emb
    bulkformer = BulkFormer(**model_params).to(DEVICE)

    ckpt_model = torch.load(CKPT_PATH, map_location="cpu")
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    bulkformer.load_state_dict(new_state_dict)
    print("[INFO] BulkFormer checkpoint loaded.")

    # --------------------------
    # High-var gene indices
    # --------------------------
    high_var_gene_idx = torch.load(HIGH_VAR_GENE_IDX_PT, map_location="cpu")
    if isinstance(high_var_gene_idx, torch.Tensor):
        high_var_gene_idx = high_var_gene_idx.numpy()
    print(f"[INFO] #high-var genes used for pooling: {len(high_var_gene_idx)}")

    # --------------------------
    # Load Phase-1 FFNN head
    # --------------------------
    print(f"[INFO] Loading Phase-1 best FFNN from: {BEST_FFN_PT}")
    ckpt_head = torch.load(BEST_FFN_PT, map_location="cpu", weights_only=False)
    hparams   = ckpt_head["hparams"]
    D_embed   = ckpt_head["meta"]["D_embed"]

    X_mean = torch.from_numpy(ckpt_head["X_mean"]).float()
    X_std  = torch.from_numpy(ckpt_head["X_std"]).float()

    head_ffnn = FFNN(
        in_dim=D_embed,
        out_dim=I,
        hidden=hparams["hidden"],
        act=hparams["act"],
        dropout=hparams["dropout"],
        batchnorm=hparams["batchnorm"],
    ).to(DEVICE)

    head_ffnn.load_state_dict(ckpt_head["state_dict"])
    print("[INFO] FFNN head loaded from Phase-1.")

    # --------------------------
    # Build fine-tuning model
    # --------------------------
    emb_mean = X_mean
    emb_std  = X_std

    model = IsoformFineTuneModel(
        bulkformer=bulkformer,
        high_var_gene_idx=high_var_gene_idx,
        head_ffnn=head_ffnn,
        emb_mean=emb_mean,
        emb_std=emb_std,
        device=DEVICE,
    ).to(DEVICE)

    # Freeze everything, then unfreeze last block + head
    model.freeze_bulkformer()
    model.unfreeze_last_block()
    for p in model.head.parameters():
        p.requires_grad = True

    # keep BatchNorm running stats fixed (batch_size = 1)
    for m in model.head.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    # Parameter groups
    encoder_params = [p for p in model.bulkformer.parameters() if p.requires_grad]
    head_params    = list(model.head.parameters())

    opt = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": head_params,    "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    # --------------------------
    # Fine-tuning loop
    # --------------------------
    best_val = math.inf
    best_state = None
    noimp = 0
    history = {"train": [], "val": []}

    print("[INFO] Starting fine-tuning...")
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS_FT + 1):
        model.train()

        # keep BatchNorm in eval mode (no running stat updates with batch_size=1)
        for m in model.head.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        total, seen = 0.0, 0

        for xb, yb in batch_iter(Xt_all, Yt_all, tr_idx_t, batch_size=BATCH_SIZE_FT, shuffle=True):
            opt.zero_grad(set_to_none=True)
            if AMP and DEVICE == "cuda":
                with torch.amp.autocast("cuda"):
                    pb = model(xb)
                    loss = criterion(pb, yb)
            else:
                pb = model(xb)
                loss = criterion(pb, yb)

            loss.backward()
            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total += float(loss.item()) * yb.size(0)
            seen  += yb.size(0)

        train_loss = total / max(1, seen)
        history["train"].append(train_loss)

        # Validation
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_mse = evaluate_on(model, Xt_all, Yt_all, va_idx_t, batch_size=BATCH_SIZE_FT)
            history["val"].append(val_mse)
            sch.step(val_mse)

            print(f"[FT] epoch {epoch:03d} | train {train_loss:.6f} | "
                  f"val {val_mse:.6f} | lr_enc={opt.param_groups[0]['lr']:.2e} "
                  f"| lr_head={opt.param_groups[1]['lr']:.2e}")

            if val_mse < best_val - 0.0:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1

            if noimp >= PATIENCE_FT:
                print("[FT] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    print(f"[INFO] Fine-tuning completed in {train_time/60:.1f} min")

    # --------------------------
    # Final evaluation
    # --------------------------
    model.eval()
    with torch.no_grad():
        tr_mse = evaluate_on(model, Xt_all, Yt_all, tr_idx_t, batch_size=BATCH_SIZE_FT)
        va_mse = evaluate_on(model, Xt_all, Yt_all, va_idx_t, batch_size=BATCH_SIZE_FT)
        te_mse = evaluate_on(model, Xt_all, Yt_all, te_idx_t, batch_size=BATCH_SIZE_FT)

        preds = []
        for xb, _ in batch_iter(Xt_all, Yt_all, te_idx_t, batch_size=BATCH_SIZE_FT, shuffle=False):
            pb = model(xb)
            preds.append(pb)
        Y_test_t = Yt_all[te_idx_t]
        Y_pred_t = torch.cat(preds, dim=0)
        te_r = pearson_mean_gpu(Y_test_t, Y_pred_t)

    print(f"[RESULT] train MSE={tr_mse:.6f} | val MSE={va_mse:.6f} | "
          f"test MSE={te_mse:.6f} | test Pearson={te_r:.4f}")

    # --------------------------
    # Save model + summary
    # --------------------------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "bulkformer_ckpt": CKPT_PATH,
            "phase1_head_pt": BEST_FFN_PT,
            "high_var_gene_idx": high_var_gene_idx,
            "train_idx": tr_idx,
            "val_idx": va_idx,
            "test_idx": te_idx,
            "metrics": {
                "train_mse": tr_mse,
                "val_mse": va_mse,
                "test_mse": te_mse,
                "test_pearson": te_r,
                "train_time_sec": train_time,
            },
        },
        OUT_PT,
    )

    summary = {
        "device": DEVICE,
        "data_pt": DATA_PT,
        "best_ffn_phase1": BEST_FFN_PT,
        "train_mse": tr_mse,
        "val_mse": va_mse,
        "test_mse": te_mse,
        "test_pearson": te_r,
        "train_time_sec": train_time,
        "batch_size": BATCH_SIZE_FT,
        "lr_encoder": LR_ENCODER,
        "lr_head": LR_HEAD,
        "max_epochs": MAX_EPOCHS_FT,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Saved fine-tuned model to:", OUT_PT)
    print("[INFO] Saved summary to:", SUMMARY_JSON)

# --------------------------
# Plot: Train vs Validation Curve
# --------------------------
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(history["train"], label="Train MSE")
    plt.plot(history["val"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Fine-Tuning: Train vs Validation MSE")
    plt.legend()
    plt.tight_layout()

    curve_path = os.path.join(RESULTS_DIR, "finetune_train_val_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved train/val curve -> {curve_path}")

except Exception as e:
    print(f"[WARN] Could not plot train/val curve: {e}")

# --------------------------
# Plot: Predicted vs True (Test Set)
# --------------------------
try:
    import matplotlib.pyplot as plt

    y_true = Y_test_t.cpu().numpy().ravel()
    y_pred = Y_pred_t.cpu().numpy().ravel()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=4, alpha=0.3)
    plt.xlabel("True log1p(isoform expression)")
    plt.ylabel("Predicted")
    plt.title(f"Prediction vs True (Pearson r={te_r:.3f})")

    minv, maxv = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], "r--", linewidth=2)

    plt.tight_layout()
    scatter_path = os.path.join(RESULTS_DIR, "finetune_pred_vs_true.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved prediction vs true plot -> {scatter_path}")

except Exception as e:
    print(f"[WARN] Could not plot pred-vs-true: {e}")

print("[DONE]")
