#!/usr/bin/env python3
"""
Phase 2: fine-tuning BulkFormer + FFNN head on my gene/isoform dataset.

- Uses original gene expression (Xg_log1p) and isoform counts (Y_tx)
- Aligns genes to the BulkFormer gene list
- Loads pretrained BulkFormer + the best FFNN head from Phase 1
- Freezes most of BulkFormer and only trains the last block + the head
"""

import os
import json
import math
import time
import random
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from collections import OrderedDict
from torch_geometric.typing import SparseTensor

from isoform_model import IsoformFineTuneModel

# ------------------------------
# basic config
# ------------------------------

SEED = 42

TEST_FRAC = 0.15
VAL_FRAC = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# keep it simple: no AMP here, it was causing NaNs
AMP = False

GRAD_CLIP = 1.0

MAX_EPOCHS_FT = 10
PATIENCE_FT = 3
EVAL_EVERY = 1
BATCH_SIZE_FT = 4  # batch_size=1 was safe with memory

# different LR for encoder vs head
LR_ENCODER = 1e-5
LR_HEAD = 5e-4

# ------------------------------
# paths
# ------------------------------

BEST_FFN_PT = "/zhome/02/d/213485/DeepIsoQ/Bulkformer/ffnn_search_bulkformer/results/best_model_bulkformer_stage2.pt"

BULKFORMER_ROOT = "/dtu/blackhole/0d/213485/s243310/BulkFormer"

GRAPH_PATH = os.path.join(BULKFORMER_ROOT, "data", "G_gtex.pt")
WEIGHTS_PATH = os.path.join(BULKFORMER_ROOT, "data", "G_gtex_weight.pt")
GENE_EMB_PATH = os.path.join(BULKFORMER_ROOT, "data", "esm2_feature_concat.pt")
CKPT_PATH = os.path.join(BULKFORMER_ROOT, "model", "Bulkformer_ckpt_epoch_29.pt")
BULKFORMER_GENE_CSV = os.path.join(BULKFORMER_ROOT, "data", "bulkformer_gene_info.csv")
HIGH_VAR_GENE_IDX_PT = os.path.join(BULKFORMER_ROOT, "data", "high_var_gene_list.pt")

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

RESULTS_DIR = "results"
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

OUT_PT = os.path.join(RESULTS_DIR, "results_phase2_finetune.pt")
SUMMARY_JSON = os.path.join(RESULTS_DIR, "finetune_summary.json")

# ------------------------------
# misc helpers
# ------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

from utils.BulkFormer import BulkFormer
from model.config import model_params


def align_to_bulkformer_genes(X_df: pd.DataFrame, gene_list):
    """
    Given X_df (samples x genes) and a BulkFormer gene_list,
    reorder/extend X_df so it matches gene_list. Missing genes get -10.
    """
    missing = list(set(gene_list) - set(X_df.columns))

    if len(missing) > 0:
        padding = pd.DataFrame(
            np.full((X_df.shape[0], len(missing)), -10, dtype=np.float32),
            columns=missing,
            index=X_df.index,
        )
        X_df = pd.concat([X_df, padding], axis=1)

    X_df = X_df[gene_list]
    return X_df, missing


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.01)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class FFNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, act="gelu", dropout=0.0, batchnorm=False):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(get_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def batch_iter(Xt, Yt, idxs, batch_size, shuffle=True):
    if shuffle:
        idxs = idxs[torch.randperm(idxs.numel(), device=idxs.device)]
    for i in range(0, idxs.numel(), batch_size):
        j = idxs[i:i + batch_size]
        yield Xt.index_select(0, j), Yt.index_select(0, j)


criterion = nn.MSELoss()


def evaluate_on(model, Xt, Yt, idxs, batch_size):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for xb, yb in batch_iter(Xt, Yt, idxs, batch_size=batch_size, shuffle=False):
            preds = model(xb)
            loss = criterion(preds, yb)

            if not torch.isfinite(loss):
                print("[EVAL] non-finite loss")
                print("  xb min/max:", xb.min().item(), xb.max().item())
                print("  yb min/max:", yb.min().item(), yb.max().item())
                print("  preds min/max:", preds.min().item(), preds.max().item())
                raise RuntimeError("Non-finite loss during evaluation")

            total_loss += float(loss.item()) * yb.size(0)
            count += yb.size(0)

    return total_loss / max(1, count)


def pearson_mean_gpu(Y_true, Y_pred):
    yt = Y_true - Y_true.mean(dim=0, keepdim=True)
    yp = Y_pred - Y_pred.mean(dim=0, keepdim=True)
    num = (yt * yp).sum(dim=0)
    den = torch.sqrt((yt * yt).sum(dim=0)) * torch.sqrt((yp * yp).sum(dim=0)) + 1e-8
    r = num / den
    return r.nanmean().item()


if __name__ == "__main__":
    print("=== PHASE 2: fine-tuning BulkFormer + FFNN head ===")
    print(f"[INFO] device: {DEVICE}")
    print(f"[INFO] loading data.pt from: {DATA_PT}")

    data = torch.load(DATA_PT, map_location="cpu", weights_only=False)

    Xg = data["Xg_log1p"].float().cpu().numpy()
    Y = torch.log1p(data["Y_tx"].float()).cpu().numpy()
    gene_ids = list(data["gene_ids"])

    N, G_orig = Xg.shape
    _, I = Y.shape

    print(f"[INFO] Xg_log1p shape = {Xg.shape}")
    print(f"[INFO] Y (log1p) shape = {Y.shape}")

    # --------------------------
    # gene alignment
    # --------------------------
    bulk_gene_info = pd.read_csv(BULKFORMER_GENE_CSV)
    bulk_gene_list = bulk_gene_info["ensg_id"].astype(str).tolist()

    print("[INFO] aligning genes to BulkFormer gene list...")
    X_df = pd.DataFrame(
        Xg,
        columns=gene_ids,
        index=[f"sample_{i}" for i in range(N)],
    )

    X_aligned_df, missing_genes = align_to_bulkformer_genes(X_df, bulk_gene_list)
    X_aligned = X_aligned_df.values.astype(np.float32)

    print(f"[INFO] aligned expression shape: {X_aligned.shape}")
    print(f"[INFO] #missing genes (imputed with -10): {len(missing_genes)}")

    # --------------------------
    # split train / val / test
    # --------------------------
    all_idx = np.arange(N)
    trval_idx, te_idx = train_test_split(
        all_idx,
        test_size=TEST_FRAC,
        random_state=SEED,
        shuffle=True,
    )
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    tr_idx, va_idx = train_test_split(
        trval_idx,
        test_size=val_rel,
        random_state=SEED,
        shuffle=True,
    )

    print(f"[INFO] split sizes: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")

    Xt_all = torch.from_numpy(X_aligned).to(DEVICE).float()
    Yt_all = torch.from_numpy(Y).to(DEVICE).float()

    tr_idx_t = torch.from_numpy(tr_idx).to(DEVICE)
    va_idx_t = torch.from_numpy(va_idx).to(DEVICE)
    te_idx_t = torch.from_numpy(te_idx).to(DEVICE)

    # --------------------------
    # BulkFormer backbone
    # --------------------------
    print("[INFO] loading BulkFormer...")

    graph_raw = torch.load(GRAPH_PATH, map_location="cpu")
    weights_raw = torch.load(WEIGHTS_PATH, map_location="cpu")
    graph = SparseTensor(
        row=graph_raw[1],
        col=graph_raw[0],
        value=weights_raw,
    ).t().to(DEVICE)

    gene_emb = torch.load(GENE_EMB_PATH, map_location="cpu")

    model_params["graph"] = graph
    model_params["gene_emb"] = gene_emb
    bulkformer = BulkFormer(**model_params).to(DEVICE)

    ckpt_model = torch.load(CKPT_PATH, map_location="cpu")
    fixed_state_dict = OrderedDict()
    for k, v in ckpt_model.items():
        new_k = k[7:] if k.startswith("module.") else k
        fixed_state_dict[new_k] = v
    bulkformer.load_state_dict(fixed_state_dict)

    print("[INFO] BulkFormer checkpoint loaded")

    # --------------------------
    # high-var genes
    # --------------------------
    high_var_gene_idx = torch.load(HIGH_VAR_GENE_IDX_PT, map_location="cpu")
    if isinstance(high_var_gene_idx, torch.Tensor):
        high_var_gene_idx = high_var_gene_idx.numpy()
    print(f"[INFO] #high-var genes used: {len(high_var_gene_idx)}")

    # --------------------------
    # Phase-1 FFNN head
    # --------------------------
    print(f"[INFO] loading Phase-1 FFNN head from: {BEST_FFN_PT}")
    ckpt_head = torch.load(BEST_FFN_PT, map_location="cpu", weights_only=False)
    hparams = ckpt_head["hparams"]
    D_embed = ckpt_head["meta"]["D_embed"]

    X_mean = torch.from_numpy(ckpt_head["X_mean"]).float()
    X_std = torch.from_numpy(ckpt_head["X_std"]).float()

    # just in case, avoid std=0
    X_std = X_std.clamp_min(1e-6)

    head_ffnn = FFNN(
        in_dim=D_embed,
        out_dim=I,
        hidden=hparams["hidden"],
        act=hparams["act"],
        dropout=hparams["dropout"],
        batchnorm=hparams["batchnorm"],
    ).to(DEVICE)

    head_ffnn.load_state_dict(ckpt_head["state_dict"])
    print("[INFO] FFNN head loaded")

    # --------------------------
    # build fine-tuning model
    # --------------------------
    model = IsoformFineTuneModel(
        bulkformer=bulkformer,
        high_var_gene_idx=high_var_gene_idx,
        head_ffnn=head_ffnn,
        emb_mean=X_mean,
        emb_std=X_std,
        device=DEVICE,
    ).to(DEVICE)

    # freeze most of BulkFormer, only last block + head are trainable
    model.freeze_bulkformer()
    model.unfreeze_last_block()
    for p in model.head.parameters():
        p.requires_grad = True

    # BatchNorm + batch_size=1 is not great, so keep BN in eval mode
    for m in model.head.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

      encoder_params = [p for p in model.bulkformer.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())

    opt = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=2
    )

    # --------------------------
    # sanity check: EPOCH 0 (no training), check if the results are the same as in the ffnn 
    # --------------------------
    print("[INFO] sanity check BEFORE fine-tuning (epoch 0)")

    # aseguramos BN en eval para el head
    for m in model.head.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    model.eval()
    with torch.no_grad():
        tr_mse_0 = evaluate_on(model, Xt_all, Yt_all, tr_idx_t, batch_size=BATCH_SIZE_FT)
        va_mse_0 = evaluate_on(model, Xt_all, Yt_all, va_idx_t, batch_size=BATCH_SIZE_FT)
        te_mse_0 = evaluate_on(model, Xt_all, Yt_all, te_idx_t, batch_size=BATCH_SIZE_FT)

    print(
        f"[EPOCH 000] train={tr_mse_0:.6f} | "
        f"val={va_mse_0:.6f} | "
        f"test={te_mse_0:.6f}"
    )

    # --------------------------
    # training loop
    # --------------------------
    print("[INFO] starting fine-tuning...")

    best_val = math.inf
    best_state = None
    no_improve = 0
    history = {"train": [], "val": []}

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS_FT + 1):
        model.train()

        # keep BN frozen
        for m in model.head.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        total_loss = 0.0
        count = 0

        for xb, yb in batch_iter(Xt_all, Yt_all, tr_idx_t, batch_size=BATCH_SIZE_FT, shuffle=True):
            opt.zero_grad(set_to_none=True)

            preds = model(xb)
            loss = criterion(preds, yb)

            if not torch.isfinite(loss):
                print("[TRAIN] non-finite loss")
                print("  xb min/max:", xb.min().item(), xb.max().item())
                print("  yb min/max:", yb.min().item(), yb.max().item())
                print("  preds min/max:", preds.min().item(), preds.max().item())
                raise RuntimeError("Non-finite loss during training")

            loss.backward()

            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            total_loss += float(loss.item()) * yb.size(0)
            count += yb.size(0)

        train_loss = total_loss / max(1, count)
        history["train"].append(train_loss)

        if epoch == 1 or epoch % EVAL_EVERY == 0:
            val_mse = evaluate_on(model, Xt_all, Yt_all, va_idx_t, batch_size=BATCH_SIZE_FT)
            history["val"].append(val_mse)
            scheduler.step(val_mse)

            print(
                f"[EPOCH {epoch:03d}] train={train_loss:.6f} | "
                f"val={val_mse:.6f} | "
                f"lr_enc={opt.param_groups[0]['lr']:.2e} | "
                f"lr_head={opt.param_groups[1]['lr']:.2e}"
            )

            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE_FT:
                print("[INFO] early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = time.time() - t0
    print(f"[INFO] finished fine-tuning in {train_time/60:.1f} min")

    # --------------------------
    # final eval
    # --------------------------
    model.eval()
    with torch.no_grad():
        tr_mse = evaluate_on(model, Xt_all, Yt_all, tr_idx_t, batch_size=BATCH_SIZE_FT)
        va_mse = evaluate_on(model, Xt_all, Yt_all, va_idx_t, batch_size=BATCH_SIZE_FT)
        te_mse = evaluate_on(model, Xt_all, Yt_all, te_idx_t, batch_size=BATCH_SIZE_FT)

        preds = []
        for xb, _ in batch_iter(Xt_all, Yt_all, te_idx_t, batch_size=BATCH_SIZE_FT, shuffle=False):
            preds.append(model(xb))

        Y_test_t = Yt_all[te_idx_t]
        Y_pred_t = torch.cat(preds, dim=0)
        te_r = pearson_mean_gpu(Y_test_t, Y_pred_t)

    print(
        f"[RESULT] train MSE={tr_mse:.6f} | "
        f"val MSE={va_mse:.6f} | "
        f"test MSE={te_mse:.6f} | "
        f"test Pearson={te_r:.4f}"
    )

    # --------------------------
    # save stuff
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
        "amp": AMP,
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] saved model to:", OUT_PT)
    print("[INFO] saved summary to:", SUMMARY_JSON)

    # --------------------------
    # simple plots
    # --------------------------
    try:
        import matplotlib.pyplot as plt

        # train vs val curve
        plt.figure(figsize=(7, 5))
        plt.plot(history["train"], label="train MSE")
        plt.plot(history["val"], label="val MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.title("fine-tuning: train vs val MSE")
        plt.legend()
        plt.tight_layout()
        curve_path = os.path.join(RESULTS_DIR, "finetune_train_val_curve.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print("[PLOT] saved train/val curve to:", curve_path)

        # scatter true vs predicted (test)
        y_true = Y_test_t.cpu().numpy().ravel()
        y_pred = Y_pred_t.cpu().numpy().ravel()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, s=4, alpha=0.3)
        plt.xlabel("true log1p(isoform expr)")
        plt.ylabel("predicted")
        plt.title(f"test set: pred vs true (r={te_r:.3f})")

        minv = min(y_true.min(), y_pred.min())
        maxv = max(y_true.max(), y_pred.max())
        plt.plot([minv, maxv], [minv, maxv], "--", linewidth=2)

        plt.tight_layout()
        scatter_path = os.path.join(RESULTS_DIR, "finetune_pred_vs_true.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print("[PLOT] saved pred-vs-true to:", scatter_path)

    except Exception as e:
        print("[WARN] could not make plots:", e)

    print("[DONE]")
