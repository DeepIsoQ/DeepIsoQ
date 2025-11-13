
import anndata as ad
import numpy as np
import torch
from scipy import sparse
import os

RUNDIR = os.path.join(os.environ["BLACKHOLE"], os.environ["USER"])
GENE_H5AD = f"{RUNDIR}/bulk_processed_genes.h5ad"
TX_H5AD   = f"{RUNDIR}/bulk_processed_transcripts.h5ad"
OUT_PT    = f"{RUNDIR}/data.pt"

print("Starting data preprocessing...")
def to_dense_f32(x):
    if hasattr(x, "to_memory"):     
        x = x.to_memory()        
    if sparse.issparse(x):
        return x.astype(np.float32).toarray()
    return np.asarray(x, dtype=np.float32)

# Load and read data (takes around 8 min)
gene_ad = ad.read_h5ad(GENE_H5AD, backed = 'r')

print(gene_ad)

tx_ad   = ad.read_h5ad(TX_H5AD, backed ='r')


print(tx_ad)

# Transform data to tensors
X_gene = torch.tensor(to_dense_f32(gene_ad.X), dtype=torch.float32)  # (N, G)
X_tx   = torch.tensor(to_dense_f32(tx_ad.X),   dtype=torch.float32)  # (N, I)

# Get gene and transcripts IDs
gene_ids = (gene_ad.var["gene_id"].astype(str).tolist()
            if "gene_id" in gene_ad.var.columns else gene_ad.var_names.astype(str).tolist())
tx_ids   = tx_ad.var_names.astype(str).tolist()

# log1p genes
Xg_log1p = torch.log1p(X_gene)
Y_tx     = X_tx  # targets = absolute isoform expression, all transcripts kept

print(f"Samples: {Xg_log1p.shape[0]}")
print(f"Genes:   {Xg_log1p.shape[1]}")
print(f"Isoforms:{Y_tx.shape[1]}")

# After loading everything, we also include some more stuff from gene_ad.uns
gene_to_transcripts   = gene_ad.uns["gene_to_transcripts"]
gene_n_transcripts    = gene_ad.uns["gene_n_transcripts"]
multi_isoform_genes   = gene_ad.uns["multi_isoform_genes"]
single_isoform_genes  = gene_ad.uns["single_isoform_genes"]
transcript_ids        = gene_ad.uns["transcript_ids"]
transcript_id_to_index = gene_ad.uns["transcript_id_to_index"]
transcript_mapping    = gene_ad.uns["transcript_mapping"]

torch.save({
    "Xg_log1p": Xg_log1p,
    "Y_tx": Y_tx,
    "gene_ids": gene_ids,
    "tx_ids": tx_ids,

    "gene_to_transcripts":   gene_to_transcripts,
    "gene_n_transcripts":    gene_n_transcripts,
    "multi_isoform_genes":   multi_isoform_genes,
    "single_isoform_genes":  single_isoform_genes,
    "transcript_ids":        transcript_ids,
    "transcript_id_to_index": transcript_id_to_index,
    "transcript_mapping":    transcript_mapping,
}, OUT_PT)



