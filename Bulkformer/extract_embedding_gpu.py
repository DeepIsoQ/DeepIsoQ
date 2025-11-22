#!/usr/bin/env python3
"""
Use pretrained BulkFormer to extract transcriptome-level embeddings
from our preprocessed bulk data (data.pt) and save them as
bulkformer_result.pt in the same BLACKHOLE/<USER> directory.

Input  (from previous preprocessing):
  - data.pt with:
      Xg_log1p : (N, G) torch.float32  (log1p gene expression)
      Y_tx     : (N, I) torch.float32  (isoform expression targets)
      gene_ids : list[str] of length G
      tx_ids   : list[str] of length I

Output:
  - bulkformer_result.pt with:
      X_bulkformer_emb : (N, D_embed) torch.float32
      Y_tx             : (N, I)       torch.float32
      gene_ids         : same as input
      tx_ids           : same as input
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from utils.BulkFormer import BulkFormer
from model.config import model_params


# ==============================
# CONFIG – EDIT THESE PATHS
# ==============================
# Where your data.pt already lives
RUNDIR = os.path.join(os.environ["BLACKHOLE"], os.environ["USER"])
IN_PT  = os.path.join(RUNDIR, "data.pt")
OUT_PT = os.path.join(RUNDIR, "bulkformer_result.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for BulkFormer resources (adapt to your repo layout)
import os
BULKFORMER_ROOT = os.path.join(os.environ["BLACKHOLE"], os.environ["USER"], "BulkFormer")

GRAPH_PATH        = os.path.join(BULKFORMER_ROOT, "data/G_gtex.pt")
WEIGHTS_PATH      = os.path.join(BULKFORMER_ROOT, "data/G_gtex_weight.pt")
GENE_EMB_PATH     = os.path.join(BULKFORMER_ROOT, "data/esm2_feature_concat.pt")
CKPT_PATH         = os.path.join(BULKFORMER_ROOT, "model/Bulkformer_ckpt_epoch_29.pt")

# Gene list used by BulkFormer (must contain the ensembl IDs in correct order)
BULKFORMER_GENE_CSV   = os.path.join(BULKFORMER_ROOT, "data/bulkformer_gene_info.csv")  # column: ensg_id
HIGH_VAR_GENE_IDX_PT  = os.path.join(BULKFORMER_ROOT, "data/high_var_gene_list.pt")     # list/array of indices

BATCH_SIZE      = 1
FEATURE_TYPE    = "transcriptome_level"   # we want per-sample embeddings
AGGREGATE_TYPE  = "all"                   # "max" | "mean" | "median" | "all"


# ==============================
# Helpers
# ==============================
def main_gene_selection(X_df: pd.DataFrame, gene_list):
    """
    Align expression matrix to BulkFormer gene list (gene_list).
    Missing genes are filled with a placeholder value -10.

    Returns:
      - X_df_full: samples x gene_list (in order)
      - to_fill_columns: list of genes that were missing and thus imputed
      - var: DataFrame with column 'mask' == 1 if gene was imputed
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

    # reorder columns to match BulkFormer gene_list exactly
    X_df_full = X_df_full[gene_list]

    var = pd.DataFrame(index=X_df_full.columns)
    var["mask"] = [1 if g in to_fill_columns else 0 for g in var.index]

    return X_df_full, to_fill_columns, var


def extract_feature(
    expr_array,
    high_var_gene_idx,
    feature_type,
    aggregate_type,
    device,
    batch_size,
    return_expr_value=False,
    esm2_emb=None,
    valid_gene_idx=None,
):
    """
    Minimal extraction function (adapted from BulkFormer notebook).

    expr_array: (N_samples, N_genes) numpy array (already in BulkFormer gene order).
    high_var_gene_idx: indices of highly variable genes (for transcriptome-level).
    feature_type: 'transcriptome_level' or 'gene_level'.
    aggregate_type: 'max' | 'mean' | 'median' | 'all' (if transcriptome_level).
    esm2_emb: required only if feature_type == 'gene_level'.
    valid_gene_idx: required only if feature_type == 'gene_level'.
    """
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    dataset = TensorDataset(expr_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_emb_list = []
    all_expr_value_list = []

    with torch.no_grad():
        if feature_type == "transcriptome_level":
            for (X,) in tqdm(loader, total=len(loader), desc="BulkFormer batches"):
                X = X.to(device)
                output, emb = model(X, [2])
                all_expr_value_list.append(output.detach().cpu().numpy())

                # emb[2]: (B, N_genes, D)
                emb_layer = emb[2].detach().cpu().numpy()
                emb_valid = emb_layer[:, high_var_gene_idx, :]  # restrict to HVGs

                if aggregate_type == "max":
                    final_emb = np.max(emb_valid, axis=1)
                elif aggregate_type == "mean":
                    final_emb = np.mean(emb_valid, axis=1)
                elif aggregate_type == "median":
                    final_emb = np.median(emb_valid, axis=1)
                elif aggregate_type == "all":
                    max_emb = np.max(emb_valid, axis=1)
                    mean_emb = np.mean(emb_valid, axis=1)
                    median_emb = np.median(emb_valid, axis=1)
                    final_emb = max_emb + mean_emb + median_emb
                else:
                    raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

                all_emb_list.append(final_emb)

            result_emb = np.vstack(all_emb_list)
            result_emb = torch.tensor(result_emb, dtype=torch.float32, device="cpu")

        elif feature_type == "gene_level":
            for (X,) in tqdm(loader, total=len(loader), desc="BulkFormer batches"):
                X = X.to(device)
                output, emb = model(X, [2])
                emb_layer = emb[2].detach().cpu().numpy()
                emb_valid = emb_layer[:, valid_gene_idx, :]
                all_emb_list.append(emb_valid)
                all_expr_value_list.append(output.detach().cpu().numpy())

            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb, dtype=torch.float32, device="cpu")
            esm2_selected = esm2_emb[valid_gene_idx]
            esm2_expanded = esm2_selected.unsqueeze(0).expand(
                all_emb_tensor.shape[0], -1, -1
            )
            esm2_expanded = esm2_expanded.to("cpu")

            result_emb = torch.cat([all_emb_tensor, esm2_expanded], dim=-1)

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    if return_expr_value:
        return np.vstack(all_expr_value_list)
    else:
        return result_emb


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print(f"[INFO] Reading input .pt from: {IN_PT}")
    data = torch.load(IN_PT, map_location="cpu")

    Xg_log1p = data["Xg_log1p"]          # (N, G) torch.float32
    Y_tx     = data["Y_tx"]              # (N, I)
    gene_ids = data["gene_ids"]          # list[str]
    tx_ids   = data["tx_ids"]            # list[str]

    N, G = Xg_log1p.shape
    print(f"[INFO] Loaded data.pt: Xg_log1p={tuple(Xg_log1p.shape)}, Y_tx={tuple(Y_tx.shape)}")

    # Turn Xg_log1p into DataFrame: samples x genes
    X_df = pd.DataFrame(
        Xg_log1p.numpy(),
        columns=list(gene_ids),
        index=[f"sample_{i}" for i in range(N)],
    )

    # ---- 1) Load BulkFormer model ----
    print("[INFO] Initializing BulkFormer...")

    graph = torch.load(GRAPH_PATH, map_location="cpu")
    weights = torch.load(WEIGHTS_PATH, map_location="cpu")
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)

    gene_emb = torch.load(GENE_EMB_PATH, map_location="cpu")

    model_params["graph"] = graph
    model_params["gene_emb"] = gene_emb
    model = BulkFormer(**model_params).to(device)

    ckpt_model = torch.load(CKPT_PATH, map_location="cpu")
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    print("[INFO] BulkFormer checkpoint loaded.")

    # ---- 2) Align our genes to BulkFormer gene list ----
    bulkformer_gene_info = pd.read_csv(BULKFORMER_GENE_CSV)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].astype(str).tolist()

    input_df, to_fill_columns, var = main_gene_selection(
        X_df=X_df, gene_list=bulkformer_gene_list
    )
    print(f"[INFO] Expression aligned to BulkFormer genes: {input_df.shape}")
    print(f"[INFO] Missing genes imputed with -10: {len(to_fill_columns)}")

    var = var.reset_index(drop=False)
    valid_gene_idx = list(var[var["mask"] == 0].index)

    high_var_gene_idx = torch.load(HIGH_VAR_GENE_IDX_PT, map_location="cpu")
    if isinstance(high_var_gene_idx, torch.Tensor):
        high_var_gene_idx = high_var_gene_idx.numpy()
    print(f"[INFO] #high-var genes used for pooling: {len(high_var_gene_idx)}")

    # ---- 3) Extract BulkFormer transcriptome-level embeddings ----
    print("[INFO] Extracting BulkFormer transcriptome-level embeddings...")
    res_bulk = extract_feature(
        expr_array=input_df.values,      # (N, N_genes_bulkformer)
        high_var_gene_idx=high_var_gene_idx,
        feature_type=FEATURE_TYPE,
        aggregate_type=AGGREGATE_TYPE,
        device=device,
        batch_size=BATCH_SIZE,
        return_expr_value=False,
        esm2_emb=model_params["gene_emb"],
        valid_gene_idx=valid_gene_idx,
    )

    print(f"[INFO] BulkFormer embedding shape: {tuple(res_bulk.shape)}")

    # ---- 4) Save result .pt for FFNN ----
    out = {
        "X_bulkformer_emb": res_bulk,   # (N, D_embed) tensor
        "Y_tx": Y_tx,                   # (N, I) tensor
        "gene_ids": gene_ids,
        "tx_ids": tx_ids,
    }
    torch.save(out, OUT_PT)
    print(f"[INFO] Saved BulkFormer result to: {OUT_PT}")
