import anndata as ad
import numpy as np
import torch

gene_data = ad.read_h5ad("/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad")
transcript_data = ad.read_h5ad("/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad")

# Extract identifiers
gene_ids = gene_data.var_names.to_list()              
tx_ids   = transcript_data.var_names.to_list()
#mapping from transcript to gene             
tx_gene_ids = transcript_data.var['gene_id'].astype(str).to_list()

X_gene = torch.tensor(gene_data.X.toarray(), dtype=torch.float32)
X_tx   = torch.tensor(transcript_data.X.toarray(), dtype=torch.float32)

#we only keep the transcripts whose gene is in the gene matrix
gene_set = set(gene_ids)
keep_tx = [gene in gene_set for gene in tx_gene_ids]
keep_tx = torch.tensor(keep_tx, dtype=torch.bool)

X_tx = X_tx[:, keep_tx]
tx_ids = [t for t, k in zip(tx_ids, keep_tx.tolist()) if k]
tx_gene_ids = [g for g, k in zip(tx_gene_ids, keep_tx.tolist()) if k]

#group the transcript by gene
grp = {}
for t, g, in zip(tx_ids, tx_gene_ids):
    if g not in grp:
        grp[g] = []
    grp[g].append(t)

isoform_ids= []
iso_sizes = []
gene_index_for_iso = []
#middle step: create isoform_ids and iso_sizes to put everything in the right order
for gene_idx, gene in enumerate(gene_ids):
    #get the list of transcripts for this gene
    lst = grp.get(gene, [])
    if not lst:
        continue
    #append list of transcripts creating an order
    isoform_ids.extend(lst)
    iso_sizes.append(len(lst))
    gene_index_for_iso.extend([gene_idx]*len(lst))

# Reorder the transcript matrix to match the new isoform order
pos = {t:i for i, t in enumerate(tx_ids)}
col_idx = torch.tensor([pos[t] for t in isoform_ids], dtype=torch.long)
Xt = X_tx[:, col_idx]

gene_index_for_iso = torch.tensor(gene_index_for_iso, dtype=torch.long)  # (I,)
G = len(gene_ids)
I = len(isoform_ids)

print(f"Number of genes: {G}, number of isoforms: {I}")

# Finally, we have a X_gene (N, G) and X_tx (N, I) matrices
# X_gene is the gene expression data that will be used as input
# X_tx is the transcript expression data that will be used as output

# targets = absolute isoform expression
Y_tpm = X_tx                    # (N, I) 

# inputs to the model: genes, log1p + z-score on train later
Xg_log1p = torch.log1p(X_gene)     # (N, G)

##################################
# Save processed data
##################################
torch.save({
    'Xg_log1p': Xg_log1p,          # (N, G)
    'Y_tpm': Y_tpm,                # (N, I)
    'isoform_ids': isoform_ids,    # list of I isoform ids
    'gene_ids': gene_ids,          # list of G gene ids
    'iso_sizes': iso_sizes,        # list of length G with number of isoforms
    'gene_index_for_iso': gene_index_for_iso,  # (I,) mapping from isoform to gene index
}, "processed_data.pt")

print("Processed data saved.")