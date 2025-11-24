#!/usr/bin/env python3
import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# ------------------------------------------------------
# Argument parsing & path resolution
# ------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect contents of the vae_latents.pt file (tensors, shapes, QC checks)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Full path to vae_latents.pt. If not given, will use $BLACKHOLE/$USER/vae_latents.pt.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="vae_latent_plots",
        help="Directory where plots and stats will be saved.",
    )
    return parser.parse_args()


def resolve_default_path():
    blackhole = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")

    if blackhole is None or user is None:
        raise RuntimeError(
            "BLACKHOLE and/or USER environment variables are not set. "
            "Set them or use --path /full/path/to/vae_latents.pt"
        )

    rundir = os.path.join(blackhole, user)
    return os.path.join(rundir, "vae_latents.pt")


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def print_stats(name, Z, outdir=None):
    Z_np = Z.cpu().numpy()
    text = []
    text.append(f"=== Stats for {name} ===")
    text.append(f"mean:       {Z_np.mean():.4f}")
    text.append(f"std:        {Z_np.std():.4f}")
    text.append(f"min / max:  {Z_np.min():.4f} / {Z_np.max():.4f}")
    text.append(f"Any NaNs?   {np.isnan(Z_np).any()}")
    text.append(f"Any Infs?   {np.isinf(Z_np).any()}")
    text.append(f"Variance per dimension (first 10 dims):")
    text.append(str(np.var(Z_np, axis=0)[:10]))

    # print to stdout
    print("\n" + "\n".join(text))

    # optionally also save to file
    if outdir is not None:
        stats_path = os.path.join(outdir, f"stats_{name}.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(text))
        print(f"[SAVED] {stats_path}")


def plot_pca(Z, title, outdir=None, filename=None):
    Z_np = Z.cpu().numpy()
    if Z_np.shape[0] < 2:
        print(f"[WARN] Not enough samples to plot PCA for {title}")
        return

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(Z_np)

    plt.figure(figsize=(5, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if outdir is not None and filename is not None:
        filepath = os.path.join(outdir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"[SAVED] {filepath}")

    plt.close()   # avoid memory buildup


def plot_latent_corr(Z, title, outdir=None, filename=None):
    Z_np = Z.cpu().numpy()
    if Z_np.shape[1] < 2:
        print(f"[WARN] Not enough dimensions to compute correlation matrix for {title}")
        return

    corr = np.corrcoef(Z_np.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.tight_layout()

    if outdir is not None and filename is not None:
        filepath = os.path.join(outdir, filename)
        plt.savefig(filepath, dpi=300)
        print(f"[SAVED] {filepath}")

    plt.close()


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    args = parse_args()

    if args.path is not None:
        pt_path = args.path
    else:
        pt_path = resolve_default_path()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n[INFO] Loading: {pt_path}")
    data = torch.load(pt_path, map_location="cpu")

    # Expect dictionary with keys "Z_train" and "Z_test"
    if "Z_train" not in data or "Z_test" not in data:
        raise KeyError("Expected keys 'Z_train' and 'Z_test' in the loaded file.")

    Z_train = data["Z_train"]
    Z_test = data["Z_test"]

    print("\n=== SHAPES ===")
    print(f"Z_train: {Z_train.shape}")   # expected (N_train, latent_dim)
    print(f"Z_test : {Z_test.shape}")    # expected (N_test, latent_dim)

    # ---- Basic stats ----
    print_stats("Z_train", Z_train, outdir=args.outdir)
    print_stats("Z_test", Z_test, outdir=args.outdir)

    # ---- PCA sanity check ----
    print("\n[INFO] Plotting PCA projections...")
    plot_pca(Z_train, "Z_train PCA",
             outdir=args.outdir, filename="pca_train.png")
    plot_pca(Z_test, "Z_test PCA",
             outdir=args.outdir, filename="pca_test.png")

    # ---- Correlation heatmap ----
    print("\n[INFO] Plotting latent correlation heatmaps...")
    plot_latent_corr(Z_train, "Latent Dimension Correlations (Train)",
                     outdir=args.outdir, filename="corr_train.png")
    plot_latent_corr(Z_test, "Latent Dimension Correlations (Test)",
                     outdir=args.outdir, filename="corr_test.png")

    print("\n[INFO] Inspection complete.")


if __name__ == "__main__":
    main()
