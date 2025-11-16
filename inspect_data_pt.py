import os
import argparse
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect contents of a data.pt file (tensors, shapes, IDs, mappings, QC checks)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Full path to data.pt. If not given, will use $BLACKHOLE/$USER/data.pt.",
    )
    return parser.parse_args()


def resolve_default_path():
    blackhole = os.environ.get("BLACKHOLE")
    user = os.environ.get("USER")

    if blackhole is None or user is None:
        raise RuntimeError(
            "BLACKHOLE and/or USER environment variables are not set. "
            "Set them or use --path /full/path/to/data.pt"
        )

    rundir = os.path.join(blackhole, user)
    return os.path.join(rundir, "data.pt")


def main():
    args = parse_args()

    if args.path is not None:
        pt_path = args.path
    else:
        pt_path = resolve_default_path()

    print(f"\nLoading: {pt_path}")
    data = torch.load(pt_path, weights_only=False)

    # -------------------------------------------------------
    # Basic structure: keys
    # -------------------------------------------------------
    print("\n=== Keys in data.pt ===")
    for k in data.keys():
        print(f" - {k}")

    # Extract tensors and metadata
    Xg = data["Xg_log1p"]
    Y = data["Y_tx"]
    gene_ids = data["gene_ids"]
    tx_ids = data["tx_ids"]
    g2t = data["gene_to_transcripts"]
    t2i = data["transcript_id_to_index"]

    # -------------------------------------------------------
    # Shapes
    # -------------------------------------------------------
    print("\n=== Tensor shapes ===")
    print("Xg_log1p (genes)      :", tuple(Xg.shape))
    print("Y_tx (isoforms)       :", tuple(Y.shape))

    # -------------------------------------------------------
    # (9) Consistency checks
    # -------------------------------------------------------
    print("\n=== Consistency checks ===")
    print("Gene count matches IDs:",
          Xg.shape[1] == len(gene_ids))
    print("Isoform count matches IDs:",
          Y.shape[1] == len(tx_ids))

    # transcript index consistency
    t2i_ok = all(t2i[tx_ids[i]] == i for i in range(len(tx_ids)))
    print("Transcript index consistency:", t2i_ok)

    # -------------------------------------------------------
    # (1) Distribution of transcripts per gene
    # -------------------------------------------------------
    print("\n=== Transcript count per gene ===")
    isoform_counts = [len(g2t[g]) for g in g2t]
    isoform_counts = np.array(isoform_counts)

    print("Genes:", len(isoform_counts))
    print("Min isoforms per gene:", isoform_counts.min())
    print("Max isoforms per gene:", isoform_counts.max())
    print("Mean isoforms per gene:", isoform_counts.mean())
    print("Genes with 1 isoform:", np.sum(isoform_counts == 1))
    print("Genes with 2 isoforms:", np.sum(isoform_counts == 2))
    print("Genes with ≥3 isoforms:", np.sum(isoform_counts >= 3))

    # -------------------------------------------------------
    # (2) NaN / Inf checks
    # -------------------------------------------------------
    print("\n=== NaN / Inf checks ===")
    print("Xg_log1p NaN:", torch.isnan(Xg).any().item())
    print("Xg_log1p Inf:", torch.isinf(Xg).any().item())
    print("Y_tx NaN:", torch.isnan(Y).any().item())
    print("Y_tx Inf:", torch.isinf(Y).any().item())

    # -------------------------------------------------------
    # (3) Sparsity statistics
    # -------------------------------------------------------
    print("\n=== Sparsity statistics ===")
    Xg_zero = (Xg == 0).sum().item()
    Y_zero = (Y == 0).sum().item()

    Xg_total = Xg.numel()
    Y_total = Y.numel()

    print(f"Xg_log1p sparsity: {Xg_zero / Xg_total:.4f} ({Xg_zero}/{Xg_total})")
    print(f"Y_tx sparsity:     {Y_zero / Y_total:.4f} ({Y_zero}/{Y_total})")

    # -------------------------------------------------------
    # (8) Genes/transcripts with zero total expression
    # -------------------------------------------------------
    print("\n=== Zero-expression genes & isoforms ===")
    zero_gene = (Xg.sum(dim=0) == 0).sum().item()
    zero_tx = (Y.sum(dim=0) == 0).sum().item()

    print("Zero-expression genes:   ", zero_gene)
    print("Zero-expression isoforms:", zero_tx)

    # -------------------------------------------------------
    # (6) Detailed transcript index consistency example
    # -------------------------------------------------------
    example_tx = tx_ids[0]
    example_index = t2i[example_tx]

    print("\n=== Transcript ID → index example ===")
    print("Transcript ID:", example_tx)
    print("Mapped index :", example_index)
    print("Vector length consistency:",
          example_index < Y.shape[1])

    print("\n=== DONE inspecting data.pt ===\n")


if __name__ == "__main__":
    main()
