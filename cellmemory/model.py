import torch
import torch.nn as nn
from torch.cuda import amp

from .transformers import TransformerEncoder
from einops import repeat

from .utils import EmbeddingLayer


class CellMemoryModel(nn.Module):
	"""CellMemory model implementation"""

	def __init__(self, config, vocab_len, num_classes):
		super().__init__()
		self.use_amp = config.use_amp
		self.batch_size = config.batch_size
		self.h_dim = config.h_dim
		self.ffn_dim = config.ffn_dim
		self.num_layers = config.num_layers
		self.num_heads = config.num_heads
		self.dropout = config.dropout
		self.max_bin = config.max_bin
		self.use_topk = config.use_topk
		self.topk = config.topk
		self.vocab_len = vocab_len
		self.mem_slots = config.mem_slots
		self.num_classes = num_classes
		self.processing = config.processing
		
		self.embedding = EmbeddingLayer(
			max_tokens=self.max_bin,
			seq_length=self.vocab_len,
			embed_dim=self.h_dim,
			dropout=self.dropout,
			processing=self.processing
		)
		
		# Transformer encoder (cross attention)
		self.transformer = TransformerEncoder(
			embed_dim=self.h_dim,
			ffn_dim=self.ffn_dim,
			num_layers=self.num_layers,
			num_heads=self.num_heads,
			dropout=self.dropout,
			share_parameters=True,
			shared_memory_attention=True,
			use_topk=self.use_topk,
			topk=self.topk,
			mem_slots=self.mem_slots,
			null_attention=False,
			num_steps=int(self.vocab_len+1),
			mode=self.processing
		)
		
		self.cls_token = nn.Parameter(torch.randn(1, 1, self.h_dim))
		
		self.mlp_head = nn.Linear(self.h_dim, self.num_classes)
		
		self.memory_mlp = nn.Linear(int(self.h_dim*self.mem_slots), self.h_dim)

	def forward(self, data_inputs, gene_idx_matrix=None, padding_matrix=None):
		
		with amp.autocast(enabled=self.use_amp):
			# Embedding
			x = self.embedding(data_inputs, gene_idx_matrix)
			
			# Add CLS token
			b = x.size(0)
			cls_tokens = repeat(self.cls_token, "() n d -> b n d", b = b)
			x = torch.cat((cls_tokens, x), dim=1)
			
			# Update padding mask for CLS token (if mode=fast)
			if padding_matrix is not None:
				padding_matrix = torch.cat((torch.zeros((b, 1), dtype=torch.bool).to(padding_matrix.device), padding_matrix), dim=1)
			
			# Transformer
			x, memory, attention_x = self.transformer(x, mask=padding_matrix)
			
			# MLP head
			y = self.mlp_head(x[:, 0])
			
			# Memory MLP (if use memory)
			memory = self.memory_mlp(memory.reshape(b, -1))
		
		return y, x[:, 0], memory, attention_x

