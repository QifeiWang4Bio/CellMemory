import os
import numpy as np
import scanpy as sc
import sklearn.metrics as metrics
import anndata as ad
import pandas as pd
import scipy.sparse
from tqdm import tqdm
from typing import Tuple, Union, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

from .utils import get_device, get_memory_score, get_tag_list

def evaluate_metrics(targets, preds):
    acc = metrics.accuracy_score(targets, preds)
    f1 = metrics.f1_score(targets, preds, average="macro")
    return acc, f1


class AttentionProcessor:
    """Process attention outputs in full or fast mode.
    
    Args:
        tag2cls: Whether to process attention outputs for cls
        mode: 'full' or 'fast'
    
    Shapes:
        - attention_in: [batch, slots, 1+tokens]
        - attention_out: [batch, 1+tokens, slots]
        - Output when tag2cls:
            - fast mode: [batch, 1+seq_len_filter]
            - full mode: [batch, 1+seq_len]
        - Output when not tag2cls:
            - fast mode: [batch, slots, 1+seq_len_filter]
            - full mode: [batch, slots, 1+seq_len]
    """
    def __init__(self, tag2cls: bool, mode: str = 'fast'):
        self.tag2cls = tag2cls
        self.mode = mode
        self.attention_outputs: List[torch.Tensor] = []
        self.gene_ids: List[torch.Tensor] = []

    def process_attention(self, attention_x: tuple, device: torch.device) -> torch.Tensor:
        attention_in, attention_out = attention_x

        if self.tag2cls:
            # Common processing for both modes
            attention_out = attention_out[:, 0, :].unsqueeze(1)
            attention = torch.bmm(
                attention_out.to(device),
                attention_in.to(device)
            )
            if self.mode == 'fast':
                return attention.reshape(attention.shape[0], attention.shape[2]).cpu()  # [batch, 1+seq_len_filter]
            return attention.squeeze(1).cpu()  # [batch, 1+seq_len]

        return attention_out.cpu()  # [batch, 1+seq_len/seq_len_filter, mem_slots]

    def update(self, attention_x: tuple, device: torch.device, gene_ids: Optional[torch.Tensor] = None):
        processed_attention = self.process_attention(attention_x, device)
        self.attention_outputs.append(processed_attention)

        if gene_ids is not None and self.mode == 'fast':
            self.gene_ids.append(gene_ids.cpu())

    def get_attention(self) -> torch.Tensor:
        """Get concatenated attention outputs."""
        return torch.cat(self.attention_outputs, dim=0)

    def get_gene_ids(self) -> Optional[torch.Tensor]:
        """Get concatenated gene ids."""
        if not self.gene_ids:
            return None
        return torch.cat(self.gene_ids, dim=0)


def train_epoch(
        net: nn.Module, 
        dataloader: DataLoader, 
        pre_loss_fn: callable,
        criterion: callable,
        optimizer: torch.optim.Optimizer,
        scaler: amp.GradScaler,
        epoch: int, 
        history_loss: list, 
        history_acc: list, 
        history_f1: list,
        device: torch.device,
        config: any
) -> Tuple[list, list, list]:
    net.train()
    total_loss = 0
    total_preds = np.zeros(len(dataloader.dataset), dtype=np.int64)
    total_targets = np.zeros(len(dataloader.dataset), dtype=np.int64)

    for batch_idx, batch_data in enumerate(tqdm(dataloader, leave=False)):
        
        batch_num = len(batch_data)
        optimizer.zero_grad()
        with amp.autocast(enabled=config.use_amp):
            if batch_num == 2:
                inputs, targets = [x.to(device) for x in batch_data]
                outputs, tokens_cls, memory, attention_x = net(inputs)
            else:
                inputs, targets, gene_ids_train, padding_mat_train = [x.to(device) for x in batch_data]
                outputs, tokens_cls, memory, attention_x = net(
                    inputs,
                    gene_ids_train,
                    padding_mat_train
                )
            # outputs, tokens_cls, memory, attention_x = net(inputs.to(device))
            outputs = pre_loss_fn(outputs)
            loss = criterion(outputs, targets)
        
        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        _, pred = outputs.max(1)
        start_idx = batch_idx*dataloader.batch_size
        end_idx = start_idx+len(targets)
        total_preds[start_idx:end_idx] = pred.cpu().numpy()
        total_targets[start_idx:end_idx] = targets.cpu().numpy()

    acc, f1_score_Macro = evaluate_metrics(total_targets, total_preds)
    del total_preds, total_targets

    avg_loss = total_loss / (batch_idx+1)
    print(
        f"\n[Epoch:{epoch+1:3d}]: "
        f"{'Train' if net.training else 'Valid'} >>>  "
        f"Loss:{avg_loss:.3f}  "
        f"Acc:{100.*acc:6.2f}  "
        f"F1:{100.*f1_score_Macro:6.2f}"
    )
    # print(f"\n[Epoch: {epoch+1}]: Train >>>  Loss: {avg_loss:.3f}  Acc: {100.*acc:.2f}  F1: {100.*f1_score_Macro:.2f}")
    history_loss.append(avg_loss)
    history_acc.append(100.*acc)
    history_f1.append(100.*f1_score_Macro)

    torch.cuda.empty_cache()

    return history_loss, history_acc, history_f1


def valid_epoch(
        net: nn.Module, 
        dataloader: DataLoader, 
        pre_loss_fn: callable,
        criterion: callable,
        best_val_loss: float,
        patience: int,
        epoch: int, 
        history_loss: list, 
        history_acc: list, 
        history_f1: list,
        dir_now: str,
        Project: str,
        device: torch.device,
        config: any
) -> Tuple[int, float, list, list, list, np.ndarray, np.ndarray]:
    net.eval()
    total_loss = 0
    total_preds = np.zeros(len(dataloader.dataset), dtype=np.int64)
    total_targets = np.zeros(len(dataloader.dataset), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, leave=False)):
            batch_num = len(batch_data)
            if batch_num == 2:
                inputs, targets = [x.to(device) for x in batch_data]
                outputs, tokens_cls, memory, attention_x = net(inputs)
            else:
                inputs, targets, gene_ids_valid, padding_mat_valid = [x.to(device) for x in batch_data]
                outputs, tokens_cls, memory, attention_x = net(
                    inputs,
                    gene_ids_valid,
                    padding_mat_valid
                )
            outputs = pre_loss_fn(outputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, pred = outputs.max(1)
            start_idx = batch_idx*dataloader.batch_size
            end_idx = start_idx+len(targets)
            total_preds[start_idx:end_idx] = pred.cpu().numpy()
            total_targets[start_idx:end_idx] = targets.cpu().numpy()
    
    acc, f1_score_Macro = evaluate_metrics(total_targets, total_preds)

    avg_loss = total_loss / (batch_idx+1)
    print(
        f"[Epoch:{epoch+1:3d}]: "
        f"{'Train' if net.training else 'Valid'} >>>  "
        f"Loss:{avg_loss:.3f}  "
        f"Acc:{100.*acc:6.2f}  "
        f"F1:{100.*f1_score_Macro:6.2f}"
    )
    # print(f"\n[Epoch: {epoch+1}]: Valid >>>  Loss: {avg_loss:.3f}  Acc: {100.*acc:.2f}  F1: {100.*f1_score_Macro:.2f}")
    history_loss.append(avg_loss)
    history_acc.append(100.*acc)
    history_f1.append(100.*f1_score_Macro)

    if avg_loss < best_val_loss:
        print("Model saved..")
        state = {
            "net": net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        checkpoint_path = os.path.join(dir_now, Project, f"{Project}_ckpt.pth")
        torch.save(state, checkpoint_path)
        best_val_loss = avg_loss
        patience = 0
    else:
        patience += 1
    
    torch.cuda.empty_cache()
     
    return patience, best_val_loss, history_loss, history_acc, history_f1, total_preds, total_targets


def predict_batch(
    net: nn.Module,
    test_loader: DataLoader,
    adata_test: ad.AnnData,
    ref_idx2celltype: dict,
    ref_gene: list,
    config: any,
    device: torch.device
) -> Union[ad.AnnData, Tuple[ad.AnnData, torch.Tensor, List[str]]]:
    """
    Batch prediction with optimized tensor operations
    """
    # 1. intilze
    novel_pred_idx = len(ref_idx2celltype)
    
    collector = PredictionCollector(
        device=device,
        cutoff=config.cutoff,
        novel_pred_idx=novel_pred_idx
    )
    
    if config.out_tag:
        attention_processor = AttentionProcessor(tag2cls=config.tag2cls, mode=config.processing)
        collector.set_attention_processor(attention_processor)

    # 2. model inference
    net.eval()
    with torch.no_grad():
        for batch_data in tqdm(test_loader, total=len(test_loader), leave=False):
            batch_num = len(batch_data)
            if batch_num > 3:
                batch_data = batch_data.to(device)
                batch_output = net(batch_data)   # : outputs, tokens_cls, memory, attention_x
                outputs, tokens_cls, memory, attention_x = batch_output
            else:
                inputs, gene_ids_test, padding_mat_test = [x.to(device) for x in batch_data]
                batch_output = net(inputs, gene_ids_test, padding_mat_test)
                outputs, tokens_cls, memory, attention_x = batch_output
            collector.update(batch_output)
            
            if config.out_tag:
                collector.attention_processor.update(
                    attention_x=attention_x,
                    device=device,
                    gene_ids=gene_ids_test if config.processing == 'fast' else None
                )

    # 3. get outputs
    query_cls = PredictionFormatter(ref_idx2celltype).format_predictions(
        collector.get_predictions(),
        collector.get_cls_outputs(),
        collector.get_pred_probs(),
        adata_test,
        novel_pred_idx
    )

    # 4. return outputs
    torch.cuda.empty_cache()
    # only return cls
    if not config.out_tag:
        return PredictionResult(cls=query_cls)
    
    attention = collector.attention_processor.get_attention()
    gene_ids = collector.attention_processor.get_gene_ids()  # if full mode, gene_ids is None

    if config.processing == 'fast':
        return interpret_fast(
            attention=attention,
            cls=query_cls,
            adata=adata_test,
            gene_ids=gene_ids,
            config=config
        )

    # Full mode
    if config.tag2cls:
        return interpret_full(
            attention=attention,
            cls=query_cls,
            adata=adata_test,
            ref_idx2celltype=ref_idx2celltype,
            ref_gene=ref_gene
            )

    # Hierarchical memory interpretation
    return PredictionResult(
        cls=query_cls,
        memory_score=np.array(attention[:, 1:, :]),  # [batch, slots, seq_len_filter]
        gene_ids=gene_ids
    )


class PredictionCollector:
    """
    Collector for model predictions and related outputs.

    Args:
        device: torch device
        cutoff: confidence threshold for predictions
        novel_pred_idx: index for novel / unassigned predictions
    """
    def __init__(
        self,
        device: torch.device,
        cutoff: float,
        novel_pred_idx: int
    ):
        self.device = device
        self.cutoff = cutoff
        self.novel_pred_idx = novel_pred_idx

        self.cls_outputs = []
        self.predictions = []
        self.pred_probs = []
        self.memory_score = []

        self.attention_processor = None

    def set_attention_processor(self, processor: AttentionProcessor):
        self.attention_processor = processor

    def update(self, batch_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """Update collector with batch outpus.

        Args:
            batch_output: tuple of (outputs, tokens_cls, memory, memory_score)
        """
        outputs, tokens_cls, memory, memory_score = batch_output

        # softmax
        pre = F.softmax(outputs, 1).cpu().detach()

        # predicition
        pred_probs, preds = torch.max(pre, dim=1)
        mask_unassign = pred_probs >= self.cutoff
        preds[~mask_unassign] = self.novel_pred_idx

        # store
        self.predictions.append(preds.numpy())
        self.pred_probs.append(pred_probs.numpy())
        self.cls_outputs.append(tokens_cls.detach().cpu())

    def get_predictions(self) -> np.ndarray:
        """Get concatenated predicitons."""
        return np.concatenate(self.predictions)

    def get_pred_probs(self) -> np.ndarray:
        """Get concatenated prediction probabilities."""
        return np.concatenate(self.pred_probs)

    def get_cls_outputs(self) -> torch.Tensor:
        """Get concatenated cls token outputs."""
        return torch.cat(self.cls_outputs, dim=0)


class PredictionFormatter:
    def __init__(self, ref_idx2celltype):
        self.ref_idx2celltype = ref_idx2celltype.copy()

    def format_predictions(
        self,
        pred_ids:np.ndarray,
        total_cls: torch.Tensor,
        total_pred_prob: np.ndarray,
        adata_test: ad.AnnData,
        novel_pred_idx: int
    ) -> ad.AnnData:
        self.ref_idx2celltype[novel_pred_idx] = "UnAssigned"

        predictions = np.array([self.ref_idx2celltype.get(idx) for idx in pred_ids])  # np.vectorize(self.ref_idx2celltype.get)(pred_ids)

        query_cls = ad.AnnData(
            total_cls.reshape(total_cls.shape[0], -1).detach().numpy()
        )

        query_cls.obs['cm_pred'] = predictions
        query_cls.obs['cm_pred_prob'] = np.round(total_pred_prob, 4)
        query_cls.obs.index = adata_test.obs.index
        query_cls.X = scipy.sparse.csr_matrix(query_cls.X)

        return query_cls


@dataclass
class PredictionResult:
    cls: ad.AnnData
    memory_score: Optional[ad.AnnData] = None
    tag: Optional[pd.DataFrame] = None
    gene_ids: Optional[torch.Tensor] = None


def interpret_fast(attention, cls, adata, gene_ids, config):
    adata_attn, tag_list = get_memory_score(
        adata=adata, 
        cls=cls, 
        total_attn=attention, 
        gene_ids=gene_ids, 
        output_path=config.Project
    )
    return PredictionResult(
        cls=cls,
        memory_score=adata_attn,
        tag=tag_list
    )


def interpret_full(attention, cls, adata, ref_idx2celltype, ref_gene):
    tag_list = get_tag_list(
        cls,
        attention[:, 1:],
        ref_idx2celltype,
        ref_gene
    )
    adata_attn = ad.AnnData(attention[:, 1:].numpy())
    adata_attn.obs = cls.obs
    adata_attn.var.index = adata.var.index
    return PredictionResult(
        cls=cls,
        memory_score=adata_attn,
        tag=tag_list
    )
