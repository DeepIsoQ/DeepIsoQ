#!/usr/bin/env python3
import os
import torch

RUNDIR = os.path.join(os.environ["BLACKHOLE"], os.environ["USER"])
DATA_PT = os.path.join(RUNDIR, "data.pt")
BF_PT   = os.path.join(RUNDIR, "bulkformer_result.pt")

print(f"[INFO] Loading: {DATA_PT}")
data = torch.load(DATA_PT, map_location="cpu")

print(f"[INFO] Loading: {BF_PT}")
bf   = torch.load(BF_PT, map_location="cpu")

Xg      = data["Xg_log1p"]      # (N, G)
Y_orig  = data["Y_tx"]          # (N, I)
gene_ids_orig = data["gene_ids"]
tx_ids_orig   = data["tx_ids"]

X_emb   = bf["X_bulkformer_emb"]  # (N, D_embed)
Y_bf    = bf["Y_tx"]
gene_ids_bf = bf["gene_ids"]
tx_ids_bf   = bf["tx_ids"]

print("\n=== SHAPES ===")
print(f"Xg_log1p          : {tuple(Xg.shape)}")
print(f"Y_tx (orig)       : {tuple(Y_orig.shape)}")
print(f"X_bulkformer_emb  : {tuple(X_emb.shape)}")
print(f"Y_tx (bf)         : {tuple(Y_bf.shape)}")

print("\n=== BASIC ASSERTS ===")
# same number of samples
assert X_emb.shape[0] == Xg.shape[0] == Y_orig.shape[0], "N mismatch between tensors"

# targets should be identical
assert Y_bf.shape == Y_orig.shape, "Y_tx shape changed"
assert torch.allclose(Y_bf, Y_orig), "Y_tx content changed!"

# IDs should be identical
assert gene_ids_bf == gene_ids_orig, "gene_ids changed!"
assert tx_ids_bf   == tx_ids_orig,   "tx_ids changed!"

print("✓ Sample count consistent")
print("✓ Y_tx identical")
print("✓ gene_ids and tx_ids identical")

print("\n=== EMBEDDING INFO ===")
N, D_embed = X_emb.shape
print(f"N samples      : {N}")
print(f"D_embed (input size for FFNN): {D_embed}")

print("\n=== VALUE CHECKS ===")
finite_mask = torch.isfinite(X_emb)
print(f"All finite? {bool(finite_mask.all())}")

print(f"Mean:      {X_emb.mean().item():.4f}")
print(f"Std:       {X_emb.std().item():.4f}")
print(f"Min / Max: {X_emb.min().item():.4f} / {X_emb.max().item():.4f}")

# just to be extra sure: compare first 5 samples by some simple stat
print("\n=== ORDERING SANITY CHECK (rough) ===")
for i in range(5):
    print(f"sample {i}:")
    print(f"  gene expr sum     = {float(Xg[i].sum()) :.4f}")
    print(f"  embedding norm    = {float(X_emb[i].norm()) :.4f}")
