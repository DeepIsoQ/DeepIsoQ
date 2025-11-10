
import anndata as ad
import numpy as np
import torch
from scipy import sparse

GENE_H5AD = "/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad"
TX_H5AD   = "/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad"


def to_dense_f32(x):
    if hasattr(x, "to_memory"):     
        x = x.to_memory()        
    if sparse.issparse(x):
        return x.astype(np.float32).toarray()
    return np.asarray(x, dtype=np.float32)

# Load and read data (takes around 8 min)
gene_ad = ad.read_h5ad(GENE_H5AD, backed = 'r')
tx_ad   = ad.read_h5ad(TX_H5AD, backed ='r')

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

# It would be possible to save the tensor so that we don't need to read the h5ad files again. 
#torch.save({
#    "Xg_log1p": Xg_log1p,   # (N, G)
#    "Y_tx": Y_tx,           # (N, I)  
#    "gene_ids": gene_ids,   # list of G
#    "tx_ids": tx_ids,       # list of I
#}, OUT_PT)


