from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
	"""Configuration for CellMemory model
    
    use_topk (bool): Use top-k selection. Default is False.
    topk (int): Number of top-k genes to select. Default is 20.
    batch_size (int): Batch size. Default is 140.
    num_epochs (int): Number of epochs. Default is 100.
    max_bin (int): Maximum number of bins. Default is 50.
    h_dim (int): Hidden dimension. Default is 256.
    ffn_dim (int): Feed-forward dimension. Default is 512.
    num_heads (int): Number of attention heads. Default is 4.
    num_layers (int): Number of encoder layers. Default is 4.
    lr (float): Learning rate. Default is 0.0003.
    mem_slots (int): Memory slots. Default is 8.
    dropout (float): Dropout rate. Default is 0.1.
    patience (int): patience for early stopping. Default is 5.
    use_amp (bool): Use Automatic mixed precision.
    seed (int): Random seed. Default is 123.

    tag2cls (bool): Whether to generate tag for cell
    out_tag (bool): Whether to output tags
    cutoff (float): Cutoff threshold
	"""

	# Model architecture
	Project: str = 'cm_output'
	max_bin: int = 50
	h_dim: int = 256
	ffn_dim: int = 512
	num_heads: int = 4
	num_layers: int = 4
	mem_slots: int = 8
	dropout: float = 0.1
	use_amp: bool = False
	max_seq_len: int = 10000

	# Training
	batch_size: int = 140
	lr: float = 0.0003
	num_epochs: int = 100
	patience: int = 5
	seed: int = 123
	exp_bin: str = 'bin_scale'
	processing: str = 'fast'

	# Features
	use_topk: bool = False
	topk: Optional[int] = 20

	# inference
	tag2cls: bool = True
	out_tag: bool = False
	cutoff: float = 0.1


	def __post_init__(self):
		"""Validate configuration"""
		assert self.h_dim % self.num_heads == 0, "Hidden dimension must be divisible by number of heads"


