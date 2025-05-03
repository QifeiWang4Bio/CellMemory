import os
import random
import numpy as np
import scanpy as sc
import pandas as pd
import time
import scipy
import anndata as ad
from scipy.sparse import csr_matrix


import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            return 'cuda:0'
        print("Using 1 GPU.")
        return 'cuda'
    print('Using CPU.')
    return 'cpu'


class EmbeddingLayer(nn.Module):
    """Embedding layer with positional encoding

    Args:
        max_tokens: Maximum number of unique tokens
        seq_length: Maximum sequence length
        embed_dim: Embedding dimension
        dropout: Dropout rate
        pad_idx: Padding index for embeddings
        processing: Whether to process fast (fast or None)
    """
    def __init__(
        self, 
        max_tokens: int, 
        seq_length: int, 
        embed_dim: int, 
        dropout: float,
        pad_idx: int = 0,
        processing: str = None
    ):
        super().__init__()
        
        self.processing = processing
        self.token_embedding = None
        self.gene_ids_embedding = None
        self.position_embedding = None

        if self.processing == 'fast':
            # Token embedding
            self.token_embedding = nn.Embedding(
                num_embeddings=max_tokens + 1,  # +2 for special tokens
                embedding_dim=embed_dim,
                padding_idx=pad_idx
            )

            # Position embedding
            self.gene_ids_embedding = nn.Embedding(
                num_embeddings=seq_length + 2,
                embedding_dim=embed_dim, 
                padding_idx=seq_length + 1
            )
        else:
            # Token embedding
            self.token_embedding = nn.Embedding(
                num_embeddings=max_tokens + 2,  # +2 for special tokens
                embedding_dim=embed_dim
            )

            # Position embedding
            self.position_embedding = nn.Embedding(
                num_embeddings=seq_length + 1,
                embedding_dim=embed_dim, 
                padding_idx=pad_idx
            )

        # Normalization and regularizatino
        self.layer_norm  = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.seq_length = seq_length

    def forward(self, tokens, gene_ids=None):
        if self.processing == 'fast':
            embeddings = self.token_embedding(tokens) + self.gene_ids_embedding(gene_ids)
        else:
            pos = torch.arange(self.seq_length, dtype=torch.long, device=tokens.device)+1
            pos = pos.unsqueeze(0).expand_as(tokens)
            embeddings = self.token_embedding(tokens) + self.position_embedding(pos)
            
        return self.dropout(self.layer_norm(embeddings))

def get_tag_list(adata_cls, attn_mat, ref_idx2celltype, ref_gene, topN=100):
    # get top attention score gene list
    gene_list = {}

    for celltype in ref_idx2celltype.values():
        mask = (adata_cls.obs['cm_pred'] == celltype)
        if not mask.any():
            continue

        attn = torch.mean(attn_mat[mask], dim=0)
        if torch.isnan(attn[0]):
            continue

        _, top_indices = torch.topk(attn, topN)
        gene_list[celltype] = list(ref_gene[top_indices.cpu()])

    return pd.DataFrame(gene_list)


def get_memory_score(adata, cls, total_attn, gene_ids, output_path, topN=100):
    # fast mode
    total_attn = total_attn.numpy()
    out_gene_ids = gene_ids.numpy()

    def make_attn_mat(adata, cls, total_attn, gene_ids):
        attn_mat = np.array(np.full((adata.shape[0], adata.shape[1]), 0), dtype=np.float32)
        for i in range(adata.shape[0]):
            temp = (gene_ids[i]-1)
            temp = np.array(temp[~(temp==adata.shape[1])], dtype=int)
            attn_mat[i, temp] = total_attn[i, 0:len(temp)]
        adata_attn = ad.AnnData(attn_mat)
        adata_attn.obs = cls.obs
        adata_attn.var.index = list(adata.var.index)
        adata_attn.X = csr_matrix(adata_attn.X)
        return adata_attn
    
    adata_attn = make_attn_mat(adata, cls, total_attn, out_gene_ids)
    adata_attn.write(f"{output_path}/adata_memoryspace_score.h5ad")

    # get top genes for each celltype
    tags = pd.DataFrame()
    for i in list(adata_attn.obs.cm_pred.unique()):
        attn = adata_attn[adata_attn.obs.cm_pred.isin([i]), :]
        attn_mean = np.mean(attn.X, axis=0)
        attn_sort = np.argsort(-attn_mean)
        gs_sort = adata_attn.var.index[np.array(attn_sort[0:topN])[0].tolist()]
        tags[i] = list(gs_sort)
    
    tags.to_csv(f"{output_path}/TAGs_top{topN}.csv")

    return adata_attn, tags

