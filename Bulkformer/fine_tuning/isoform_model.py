#!/usr/bin/env python3
import torch
import torch.nn as nn


class IsoformFineTuneModel(nn.Module):
    """
    BulkFormer encoder + FFNN head for isoform prediction.

    Pipeline:
      - run BulkFormer on full gene expression
      - select high-variance genes
      - pool (max + mean + median) over those genes
      - normalize using Phase-1 embedding stats (X_mean, X_std)
      - pass through the FFNN head
    """

    def __init__(self, bulkformer, high_var_gene_idx, head_ffnn,
                 emb_mean, emb_std, device=None):
        super().__init__()

        # BulkFormer backbone
        self.bulkformer = bulkformer

        # Indices of high-variance genes (into BulkFormer gene axis)
        self.high_var_gene_idx = torch.as_tensor(
            high_var_gene_idx, dtype=torch.long
        )

        # FFNN head trained in Phase-1 on BulkFormer embeddings
        self.head = head_ffnn

        # Mean / std used to normalize embeddings in Phase-1
        # shape: (1, D_embed)
        self.emb_mean = emb_mean.view(1, -1)
        self.emb_std = emb_std.view(1, -1)

        if device is not None:
            self.to(device)

    def freeze_bulkformer(self):
        """
        Freeze all BulkFormer parameters.
        """
        for p in self.bulkformer.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        """
        Unfreeze only the last GBFormer block.

        Assumes BulkFormer has an attribute `gb_formers` that
        is a list / ModuleList of transformer blocks.
        """
        for p in self.bulkformer.gb_formers[-1].parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        x: (batch_size, n_genes) gene expression tensor, aligned
           to the BulkFormer gene order and preprocessed as in
           the original BulkFormer training.
        """
        device = x.device
        hv_idx = self.high_var_gene_idx.to(device)

        # -------------------------
        # 1) BulkFormer forward
        # -------------------------
        # We only care about the hidden representation at layer 2.
        # `hidden` is indexed by layer id; recon is ignored here.
        _, hidden = self.bulkformer(x, repr_layers=[2])
        h = hidden[2]                      # (B, n_genes, d_model)

        # -------------------------
        # 2) Pool over HVGs
        # -------------------------
        h = h[:, hv_idx, :]                # (B, n_hvg, d_model)

        h_max = h.max(dim=1).values        # (B, d_model)
        h_mean = h.mean(dim=1)             # (B, d_model)
        h_med = h.median(dim=1).values     # (B, d_model)

        # Combined embedding (same as Phase-1 extractor)
        z = h_max + h_mean + h_med         # (B, d_model)

        # -------------------------
        # 3) Normalize embedding
        # -------------------------
        emb_mean = self.emb_mean.to(device)
        emb_std = self.emb_std.to(device)

        z_norm = (z - emb_mean) / (emb_std + 1e-8)

        # -------------------------
        # 4) FFNN head
        # -------------------------
        return self.head(z_norm)
