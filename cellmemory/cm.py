import os
import random
import numpy as np
import scanpy as sc
import sklearn.metrics as metrics
import pandas as pd
import time
import matplotlib.pyplot as plt
import warnings
import anndata as ad
from tqdm import tqdm
from typing import Tuple, Union, List, Optional

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda import amp

from .dataset import training_load, inference_load
from .utils import get_device
from .plot import plot_history, plot_conf_mat
from .config import ModelConfig
from .model import CellMemoryModel
from .process import train_epoch, valid_epoch, predict_batch


def seed_everything(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def train(
	anndata: Union[str, ad.AnnData],
	Project: str,
	Label: str,
    mode: str = 'fast',
	batch_size: int = 140,
	valid_id: Union[None, int] = None,
	exp_bin: str = 'bin_scale',
	use_amp: bool = False,
    max_seq_len: int = 10000,
	seed: int = 123,
):
    """Training process of CellMemory

        anndata (Union[str, ad.AnnData]): Input data in AnnData format or file path.
        Project (str): Project name.
        Label (str): Label for the cell types.
        mode (str): Processing mode. Default is 'fast'. ['fast', 'full']
        batch_size (int): Batch size. Default is 140.
        valid_id (Union[None, int]): Validation set ID. Default is None.
        epochs (int): Number of epochs. Default is 100.
        exp_bin (str): Binning method. Default is 'scale_bin'. ['bin_scale', 'bin_norm']
        use_amp (bool): Use Automatic mixed precision.
        max_seq_len (int): Maximum sequence length. Default is 10000.
        seed (int): Random seed. Default is 123.

    """

    # 1. load config
    config = ModelConfig(
        Project=Project,
    	batch_size=batch_size,
    	exp_bin=exp_bin,
        processing=mode,
    	use_amp=use_amp,
        max_seq_len=max_seq_len,
    	seed=seed
	)

    dir_now = os.getcwd()
    seed_everything(seed = seed)
    warnings.filterwarnings("ignore")

    print('# >> Forming intuition for cells.. << #')
    print('# *********************************** #')
    print('Parameter setting: -Project',Project,'-Label',Label,'-batch_size',batch_size,'-exp_bin',exp_bin,'-use_amp',use_amp,'-seed',seed)


    # 2. load and process data
    trainloader, validloader, vocab_len, num_classes, idx2celltype = \
    	training_load(
    		anndata, 
    		Project, 
    		Label, 
    		config, 
    		valid_id
		)


    # 3. load model
    net = CellMemoryModel(config, vocab_len, num_classes)

    # TODO
    device = get_device()
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    pre_loss_fn = nn.Identity()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scaler = amp.GradScaler(enabled=config.use_amp)
    

    # 4. train model
    start_epoch = 0
    patience = 0
    train_loss_history = []
    train_acc_history = []
    train_f1_history = []
    valid_loss_history = []
    valid_acc_history = []
    valid_f1_history = []
    best_val_loss = float("inf")
    T1 = time.time()
    print("CellMemory is delineating cell representations..")
    
    for epoch in range(start_epoch, start_epoch+100):
        t1 = time.time()
        
        # train set
        train_loss_history, train_acc_history, train_f1_history = \
        	train_epoch(
                net=net, 
                dataloader=trainloader, 
                pre_loss_fn=pre_loss_fn,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch, 
                history_loss=train_loss_history, 
                history_acc=train_acc_history,
                history_f1=train_f1_history,
                device=device,
                config=config
        )
        
        # valid set
        patience, best_val_loss, valid_loss_history, valid_acc_history, valid_f1_history, total_targets, total_preds = \
            valid_epoch(
                    net=net, 
                    dataloader=validloader, 
                    pre_loss_fn=pre_loss_fn,
                    criterion=criterion,
                    best_val_loss=best_val_loss,
                    patience=patience,
                    epoch=epoch, 
                    history_loss=valid_loss_history, 
                    history_acc=valid_acc_history, 
                    history_f1=valid_f1_history, 
                    dir_now=dir_now,
                    Project=Project,
                    device=device,
                    config=config
            )

        t2 = time.time()
        print("Run time: %.d s" % (t2 - t1))
        if patience == config.patience:
            print("\nEarly stopping")
            break
        elif patience == 0:
            out_targets, out_preds = total_targets, total_preds
        scheduler.step()

    # ***************************** #
    # >>>   history plotting    <<< #
    # ***************************** #
    T2 = time.time()
    print("Total Training time: %.d s" % (T2 - T1))
    print("\nPlotting training records..")
    
    plt.rcParams["pdf.fonttype"]=42
    # loss
    plot_history(
            train_loss_history, 
            valid_loss_history, 
            "Loss", 
            dir_now+"/"+Project+"/fig_loss.pdf"
    )
    # confusion matrix
    plot_conf_mat(
            out_targets, 
            out_preds, 
            output=dir_now+"/"+Project+"/fig_conf_mat_valid.pdf",
            idx2celltype=idx2celltype
    )


def generate(
	anndata: Union[str, ad.AnnData],
	Project: str,
	out_tag: bool = False,
	tag2cls: bool = True,
	cutoff: float = 0.1,
	batch_size: int = 140,
	exp_bin: str = 'bin_scale',
    mode: str = 'fast',
    max_seq_len: int = 10000,
	use_amp: bool = False,
	seed: int = 123,
):
    """Predict / Inference process of CellMemory

        anndata (Union[str, ad.AnnData]): Input data in AnnData format or file path.
        Project (str): Project name.
        out_tag (bool):
        tag2cls (bool): 
        cutoff (float):
        batch_size (int): Batch size. Default is 140.
        exp_bin (str): Binning method. Default is 'scale_bin'. ['bin_scale', 'bin_norm']
        mode (str): Processing mode. Default is 'fast'. ['fast', 'full']
        max_seq_len (int): Maximum sequence length. Default is 10000.
        use_amp (bool): Use Automatic mixed precision.
        seed (int): Random seed. Default is 123.

    """
    seed_everything(seed = seed)
    warnings.filterwarnings ("ignore")

    # 1. load config
    config = ModelConfig(
        Project=Project,
    	batch_size=batch_size,
    	exp_bin=exp_bin,
        processing=mode,
    	use_amp=use_amp,
        tag2cls=tag2cls,
        out_tag=out_tag,
        cutoff=cutoff,
        max_seq_len=max_seq_len,
    	seed=seed
	)

    # 2. load and process data
    testloader, adata_test, vocab_len, ref_num_classes, ref_idx2celltype, ref_gene = \
    	inference_load(
    		anndata, 
    		Project, 
    		config
		)

    # 3. load model
    net = CellMemoryModel(config, vocab_len, ref_num_classes)

    print("==> Resuming from checkpoint..")
    
    device = get_device()
    net = net.to(device)
    checkpoint = torch.load(Project+'/'+Project+'_ckpt.pth', map_location=torch.device(device))
    if torch.cuda.device_count() > 1:
        net.load_state_dict(checkpoint["net"])
        net = nn.DataParallel(net)
    else:
        net.load_state_dict(checkpoint["net"])

    # 4. generate CLS and cell identify
    print("CellMemory is deciphering query cells..")
    output = predict_batch(
                net=net,
                test_loader=testloader,
                adata_test=adata_test,
                ref_idx2celltype=ref_idx2celltype,
                ref_gene=ref_gene,
                config=config,
                device=device
            )

    return output




