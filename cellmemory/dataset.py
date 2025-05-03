import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as Data

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn.metrics as metrics
import anndata as ad
from typing import Union, Optional, Tuple, Dict, Any


import os
import random
from pathlib import Path

from .utils import get_device

def training_load(
    anndata: Union[str, ad.AnnData],
    Project: str,
    Label: str,
    config: Any,
    valid_id: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, int, int, Dict]:
    
    # 1. Read data
    print("Reading Data:", Project)
    adata = read_anndata(anndata)

    celltype_total = np.array(adata.obs[Label])

    # 2. Split train set and valid set
    if valid_id == None:
        # random 
        adata_train, adata_valid, celltype_train_idx, celltype_valid_idx, idx2celltype = \
            Split_Train_Valid_Dataset(
                adata, 
                celltype_total
            )
    else:
        # Selecting which fold as the validation set
        adata_train, adata_valid, celltype_train_idx, celltype_valid_idx, idx2celltype = \
            Split_Train_Valid_Dataset_(
                adata, 
                celltype_total, 
                valid_id
            )
    
    # 3. matrix todense
    adata_train_mat = mat_todense(adata_train)
    adata_valid_mat = mat_todense(adata_valid)

    # 4. expression value processing. up bin or scale bin
    adata_train_mat, adata_valid_mat, max_bin_train = value_process(adata_train_mat, 
                                                    adata_valid_mat, 
                                                    config.exp_bin, 
                                                    config.max_bin,
                                                )
    # 4.1 if processing is fast, cells need filter zero exp gene
    if config.processing == 'fast':
        adata_train_mat, adata_valid_mat, gene_ids_train, gene_ids_valid, gene_mappings = \
            tokenize_padding(
                adata=adata,
                max_seq_len=config.max_seq_len,
                adata_train_mat=adata_train_mat, 
                adata_valid_mat=adata_valid_mat, 
                is_training=True, 
            )


    # 5. output path
    project_dir = Path(Project)
    project_dir.mkdir(exist_ok=True)

    # save metadata files
    project_dir.joinpath(f"{Project}_ref_gene.txt").write_text(
        pd.DataFrame(adata.var.index).to_csv(header=0, index=0)
    )
    project_dir.joinpath(f"{Project}_idx2celltype.txt").write_text(
        pd.DataFrame(list(idx2celltype.values())).to_csv(header=0, index=0)
    )
    project_dir.joinpath("MaxBin.txt").write_text(str(max_bin_train))

    # 6. To tensor
    if config.processing == 'fast':
        adata_train_mat, adata_valid_mat, celltype_train_idx, celltype_valid_idx, \
            vocab_len, num_classes, gene_ids_train, gene_ids_valid, \
                padding_mat_train, padding_mat_valid \
            = toTensor(
                adata=adata, 
                adata_train_mat=adata_train_mat, 
                adata_valid_mat=adata_valid_mat, 
                celltype_train_idx=celltype_train_idx, 
                celltype_valid_idx=celltype_valid_idx, 
                celltype_total=celltype_total,
                gene_ids_train=gene_ids_train,
                gene_ids_valid=gene_ids_valid,
                gene_id_dic=gene_mappings['gene_to_id']
            )
    else:
        adata_train_mat, adata_valid_mat, celltype_train_idx, celltype_valid_idx, \
            vocab_len, num_classes \
            = toTensor(
                adata=adata, 
                adata_train_mat=adata_train_mat, 
                adata_valid_mat=adata_valid_mat, 
                celltype_train_idx=celltype_train_idx, 
                celltype_valid_idx=celltype_valid_idx, 
                celltype_total=celltype_total
            )

    # 7. Build data loader
    if config.processing == 'fast':
        trainloader = data_loader(
                    adata=adata_train_mat, 
                    celltype_idx=celltype_train_idx, 
                    gene_ids=gene_ids_train,
                    padding_mat=padding_mat_train,
                    batch_size=config.batch_size, 
                    shuffle=True
                )
        validloader = data_loader(
                    adata=adata_valid_mat, 
                    celltype_idx=celltype_valid_idx, 
                    gene_ids=gene_ids_valid,
                    padding_mat=padding_mat_valid,
                    batch_size=config.batch_size, 
                    shuffle=True
                )
    else:
        trainloader = data_loader(
                    adata=adata_train_mat, 
                    celltype_idx=celltype_train_idx, 
                    batch_size=config.batch_size, 
                    shuffle=True
                )
        validloader = data_loader(
                    adata=adata_valid_mat, 
                    celltype_idx=celltype_valid_idx, 
                    batch_size=config.batch_size, 
                    shuffle=True
                )

    return trainloader, validloader, vocab_len, num_classes, idx2celltype



def inference_load(
    anndata: Union[str, ad.AnnData],
    Project: str,
    config
):
    
    # 1. Read data
    print(f"\nLoading Model from: {Project}")
    if not os.path.exists(Project):
        raise FileNotFoundError(f"Project directory {Project} does not exist!")
    
    adata_test = read_anndata(anndata)

    # 2. load ref metadata
    ref_gene = np.array(pd.read_csv(f"{Project}/{Project}_ref_gene.txt", header=None)[0], dtype=str)
    cell_id = pd.read_csv(f"{Project}/{Project}_idx2celltype.txt", header=None)[0].tolist()
    max_bin = int(np.loadtxt(f"{Project}/MaxBin.txt"))

    idx = list(range(0, len(cell_id)))
    celltype2idx = dict(zip(cell_id, idx))
    ref_idx2celltype = {i: w for i,w in enumerate(celltype2idx)}
    ref_num_classes = len(ref_idx2celltype)

    # 3. mat to dense
    adata_test_mat = mat_todense(adata_test)
    
    # 4. expression value processing. up bin or scale bin
    if 'bin_norm' in config.exp_bin:
        query_max_bin = int(np.ceil(np.max(adata_test_mat)))
        adata_test_mat = ToBins(adata_test_mat, max_bin, "up")
        adata_test_mat[adata_test_mat > max_bin] = max_bin
    elif 'bin_scale' in config.exp_bin:
        query_max_bin = int(max_bin)
        adata_test_mat = ScaleBins(adata_test_mat, max_bin)

    print("Reference Categories:", len(ref_idx2celltype))
    print("Query set:", adata_test_mat.shape)
    print("Query Max bins:", query_max_bin)
    print("Ref Max bins:", max_bin)

    # if processing is fast, cells need filter zero exp gene
    if config.processing == 'fast':
        adata_test_mat, gene_ids_test, gene_mappings = \
            tokenize_padding(
                adata=adata_test,
                max_seq_len=config.max_seq_len,
                adata_test_mat=adata_test_mat, 
                is_training=False, 
            )

    # 5. to tensor
    if config.processing == 'fast':
        adata_test_mat = torch.LongTensor(adata_test_mat)
        gene_ids_test = torch.LongTensor(gene_ids_test)
        padding_mat_test = gene_ids_test.eq(gene_mappings['gene_to_id']['<pad>'])
    else:
        adata_test_mat = torch.LongTensor(adata_test_mat)

    # 6. build data loader
    if config.processing == 'fast':
        testloader = data_loader(
                adata=adata_test_mat, 
                gene_ids=gene_ids_test,
                padding_mat=padding_mat_test,
                batch_size=config.batch_size,
                shuffle=False
            )
    else:
        testloader = data_loader(
                adata=adata_test_mat, 
                batch_size=config.batch_size,
                shuffle=False
            )

    vocab_len = len(ref_gene)

    return testloader, adata_test, vocab_len, ref_num_classes, ref_idx2celltype, ref_gene


def read_anndata(anndata):
    if isinstance(anndata, str):
        if anndata.endswith("h5ad"):
            adata = sc.read_h5ad(anndata)
        elif anndata.endswith("h5"):
            adata = sc.read_10x_h5(anndata)
        else:
            raise ValueError("Pleace check the input file format!")
    else: 
        adata = anndata
    return adata


def split_train_valid(cell_num, ratio):
    # 80% train and 20% validation dataset
    test_num = round(cell_num*ratio)
    train_num = cell_num - test_num
    random.seed(123)
    test = random.sample(range(0,cell_num), test_num)
    train = list(set(range(0,cell_num)).symmetric_difference(test))
    return train, test


def celltypeToidx(
    celltype_total, 
    celltype_train=None, 
    celltype_valid=None
):
    # celltype to idx
    celltype = list(np.unique(celltype_total))
    idx = list(range(0, len(celltype)))
    celltype2idx = dict(zip(celltype, idx))
    idx2celltype = {i: w for i,w in enumerate(celltype2idx)}
    
    celltype_train_idx = []
    for i in range(len(celltype_train)):
        celltype_train_idx.append(celltype2idx[celltype_train[i]])
     
    celltype_valid_idx = []
    for i in range(len(celltype_valid)):
        celltype_valid_idx.append(celltype2idx[celltype_valid[i]])

    return celltype_train_idx, celltype_valid_idx, idx2celltype


def Split_Train_Valid_Dataset(adata, celltype_total):
    train_set, valid_set = split_train_valid(adata.shape[0], 0.2)
    adata_train, celltype_train = adata[train_set, :], celltype_total[train_set]
    adata_valid, celltype_valid = adata[valid_set, :], celltype_total[valid_set]
    celltype_train_idx, celltype_valid_idx, idx2celltype = celltypeToidx(
                                                                celltype_total, 
                                                                celltype_train, 
                                                                celltype_valid
                                                            )
    return adata_train, adata_valid, celltype_train_idx, celltype_valid_idx, idx2celltype


def Split_Train_Valid_Dataset_(adata, celltype_total, valid_id):
    # valid_id: [0, 1, 2, 3, 4]   # five cross fold
    valid_start, valid_end = valid_id*0.2, (valid_id*0.2+0.2)
    valid_set = list(range(int(valid_start*adata.shape[0]), int(valid_end*adata.shape[0])))
    train_set = list(set(list(range(0, adata.shape[0]))) - set(valid_set))
    adata_train, celltype_train = adata[train_set, :], celltype_total[train_set]
    adata_valid, celltype_valid = adata[valid_set, :], celltype_total[valid_set]
    celltype_train_idx, celltype_valid_idx, idx2celltype = celltypeToidx(
                                                                celltype_total, 
                                                                celltype_train, 
                                                                celltype_valid
                                                            )
    return adata_train, adata_valid, celltype_train_idx, celltype_valid_idx, idx2celltype


def ToBins(continuous_tokens, max_limit, direction):
    if direction == 'down':
        bin_tokens = np.floor(continuous_tokens)
    else:
        bin_tokens = np.ceil(continuous_tokens)
    bin_tokens[bin_tokens>max_limit] = max_limit
    return bin_tokens


def ScaleBins(continuous_tokens, max_limit):
    for i in range(continuous_tokens.shape[0]):
        token_raw = continuous_tokens[i,:]
        
        non_zero_pos = token_raw.nonzero()[0]
        non_zero_values = token_raw[non_zero_pos]
        
        bins = np.quantile(np.array(non_zero_values), np.linspace(0, 1, max_limit))
        bin_values = np.digitize(non_zero_values, bins)

        continuous_tokens[i, non_zero_pos] = bin_values

    return continuous_tokens


def mat_todense(adata):
    if scipy.sparse.issparse(adata.X):
        adata_mat = adata.X.todense().A
    else:
        adata_mat = adata.X.A
    return adata_mat


def value_process(adata_train_mat, adata_valid_mat, exp_bin, max_bin):
    if exp_bin == 'bin_norm':
        max_bin = int(
            np.ceil(max(np.max(adata_train_mat), np.max(adata_valid_mat)))
        )
        adata_train_mat = ToBins(adata_train_mat, max_bin, "up")
        adata_valid_mat = ToBins(adata_valid_mat, max_bin, "up")
    elif exp_bin =='bin_scale':
        adata_train_mat = ScaleBins(adata_train_mat, max_bin)
        adata_valid_mat = ScaleBins(adata_valid_mat, max_bin)

    print("Train set:", adata_train_mat.shape)
    print("Valid set:", adata_valid_mat.shape)
    print("Max bins:", max_bin)

    return adata_train_mat, adata_valid_mat, max_bin


def toTensor(adata, adata_train_mat, adata_valid_mat, celltype_train_idx, celltype_valid_idx, celltype_total,
             gene_ids_train=None, gene_ids_valid=None, gene_id_dic=None
             ) -> Tuple:
    """Convert data matrices and indices to PyTorch tensors.
    
    Args:
        adata: ad.AnnData
        adata_train_mat: Training data matrix
        adata_valid_mat: Validation data matrix
        celltype_train_idx: Training cell type indices
        celltype_valid_idx: Validation cell type indices
        celltype_total: Total cell types
        gene_ids_train: Optinal gene IDs for training data (for fast processing mode)
        gene_ids_valid: Optinal gene IDs for validation data (for fast processing mode)
        gene_id_dic: Optinal gene ID dictionary (for fast processing mode)
    
    Returns:
        Tuple containing:
        - adata_train_mat: Tensor of training data
        - adata_valid_mat: Tensor of validation data
        - celltype_train_idx: Tensor of training cell type indices
        - celltype_valid_idx: Tensor of validation cell type indices
        - vocab_len: Vocabulary length
        - num_classes: Number of classes
        - gene_ids_train: Tensor of training gene IDs (if in fast mode)
        - gene_ids_valid: Tensor of validation gene IDs (if in fast mode)
        - padding_mat_train: Training padding mask (if in fast mode)
        - padding_mat_valid: Validation padding mask (if in fast mode)
    """
    # 1. Convert data matrices to PyTorch tensors
    adata_train_mat = torch.LongTensor(adata_train_mat)
    adata_valid_mat = torch.LongTensor(adata_valid_mat)
    celltype_train_idx = torch.LongTensor(celltype_train_idx)
    celltype_valid_idx = torch.LongTensor(celltype_valid_idx)
    
    num_classes = len(np.unique(celltype_total))
    vocab_len = adata.shape[1]

    print("Learning Categories:", num_classes)

    # Hanle gene IDs if provided (fast processing mode)
    if gene_ids_train is not None and gene_ids_valid is not None and gene_id_dic is not None:
        gene_ids_train = torch.LongTensor(gene_ids_train)
        gene_ids_valid = torch.LongTensor(gene_ids_valid)
        PAD_TOKEN = '<pad>'
        padding_mat_train = gene_ids_train.eq(gene_id_dic[PAD_TOKEN])
        padding_mat_valid = gene_ids_valid.eq(gene_id_dic[PAD_TOKEN])
        
        return (adata_train_mat, adata_valid_mat, celltype_train_idx, celltype_valid_idx, 
                vocab_len, num_classes, gene_ids_train, gene_ids_valid, 
                padding_mat_train, padding_mat_valid)
    
    return (adata_train_mat, adata_valid_mat, celltype_train_idx, celltype_valid_idx, 
            vocab_len, num_classes)


def data_loader(adata, batch_size, celltype_idx=None, gene_ids=None, padding_mat=None, shuffle=False):
    loader = torch.utils.data.DataLoader(
            dataset=get_dataset(
                tokens=adata, 
                tokens_celltype=celltype_idx, 
                gene_ids=gene_ids, 
                padding_mat=padding_mat
            ),
            batch_size=batch_size,
            num_workers=4,
            shuffle=shuffle,
            pin_memory=True
    )
    return loader


class get_dataset(Data.Dataset):
    """Dataset class for CellMemory that handles both training and testing data.
    
    Args:
    tokens: Input token data
    tokens_celltype: Optional cell type labels. Default is None for test set.
    gene_ids: Optional gene IDs for fast mode (default: None)
    padding_mat: Optional padding matrix for fast mode (default: None)
    """
    def __init__(
            self, 
            tokens: torch.Tensor,
            tokens_celltype: Optional[torch.Tensor] = None,
            gene_ids: Optional[torch.Tensor] = None,
            padding_mat: Optional[torch.Tensor] = None
        ):
        self.tokens = tokens
        self.tokens_celltype = tokens_celltype
        self.gene_ids = gene_ids
        self.padding_mat = padding_mat

        self.is_fast_mode = gene_ids is not None and padding_mat is not None
        self.is_train = tokens_celltype is not None
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        if self.is_train:
            if self.is_fast_mode:
                return (
                    self.tokens[idx],
                    self.tokens_celltype[idx],
                    self.gene_ids[idx],
                    self.padding_mat[idx]
                )
            return self.tokens[idx], self.tokens_celltype[idx]
        else:  # test mode
            if self.is_fast_mode:
                return self.tokens[idx], self.gene_ids[idx], self.padding_mat[idx]
            return self.tokens[idx]


def token_pad(mat, max_len, pad_id):
    """ Pad token matrix and generate corresponding gene ids matrix 
    Args:
        mat: input matrix of shape (n_cells, n_genes)
        max_len: maximum length after padding
        pad_id: padding ID for gene_ids_matrix

    Returns:
        tuple: (token_matrix, gene_ids_matrix)
    """
    token_mat = np.full((mat.shape[0], max_len), 0)          # token padding value > 0
    gene_ids_mat = np.full((mat.shape[0], max_len), pad_id)  # gene name padding > gene_id_dic['<pad>']
    for i in range(mat.shape[0]):
        pos_non_zero = np.nonzero(mat[i, :])[0]
        if len(pos_non_zero) <= max_len:
            token_mat[i, 0:len(pos_non_zero)] = mat[i, pos_non_zero]
            gene_ids_mat[i, 0:len(pos_non_zero)] = pos_non_zero+1
        else:
            pos_non_zero = np.sort(np.random.choice(pos_non_zero, max_len, replace=False))
            token_mat[i, 0:len(pos_non_zero)] = mat[i, pos_non_zero]
            gene_ids_mat[i, 0:len(pos_non_zero)] = pos_non_zero+1
    return token_mat, gene_ids_mat


def tokenize_padding(
        adata: ad.AnnData,
        max_seq_len: int,
        adata_train_mat: Optional[np.ndarray] = None,
        adata_valid_mat: Optional[np.ndarray] = None,
        adata_test_mat: Optional[np.ndarray] = None,
        is_training: bool = True
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict],
        Tuple[np.ndarray, np.ndarray, Dict]
    ]:
    """Process data in fast mode with tokenziation and padding.
    
    Args:
        adata: Original AnnData object containing gene information
        max_seq_len: Maximum sequence length to use 
        adata_train_mat: Training data matrix (required for training mode)
        adata_valid_mat: Validation data matrix (required for training mode)
        adata_test_mat: Testing data matrix (required for testing mode)
        is_training: Whether the data is for training or testing
    
    Returns:
        Training mode:
            - Processed training matrix
            - Processed validation matrix
            - Training matrix gene IDs
            - Validation matrix gene IDs
            - Gene mappings dictionary
        Testing mode:
            - Processed testing matrix
            - Testing matrix gene IDs
            - Gene mappings dictionary
    """
    # tokenize padding
    gene_name = list(adata.var.index)
    gene_mappings = {
        'id_to_gene': {i+1: gene for i, gene in enumerate(gene_name)},
        'gene_to_id': {gene: i+1 for i, gene in enumerate(gene_name)}
    }
    # Add paddding token
    pad_id = len(gene_name) + 1
    gene_mappings['id_to_gene'][pad_id] = '<pad>'
    gene_mappings['gene_to_id']['<pad>'] = pad_id
    
    if is_training:
        # Process sequences 
        max_len_train = min(max_seq_len, max(np.count_nonzero(adata_train_mat, axis=1)))
        max_len_valid = min(max_seq_len, max(np.count_nonzero(adata_valid_mat, axis=1)))

        # Apply padding
        adata_train_mat, gene_ids_train = token_pad(adata_train_mat, max_len_train, pad_id)
        adata_valid_mat, gene_ids_valid = token_pad(adata_valid_mat, max_len_valid, pad_id)

        print("Processed Train set:", adata_train_mat.shape)
        print("Processed Valid set:", adata_valid_mat.shape)

        return adata_train_mat, adata_valid_mat, gene_ids_train, gene_ids_valid, gene_mappings
    else:
        # Testing mode: process test data
        # Process test sequences
        max_len_test = min(max_seq_len, max(np.count_nonzero(adata_test_mat, axis=1)))

        # Apply padding
        adata_test_mat, gene_ids_test = token_pad(adata_test_mat, max_len_test, pad_id)

        print("Processed Test set:", adata_test_mat.shape)

        return adata_test_mat, gene_ids_test, gene_mappings

