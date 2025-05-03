import torch
import torch.nn as nn
import types
import math
import numpy as np

args = types.SimpleNamespace()
args.use_module_communication = 'true'
args.encoder_embed_dim = 512
args.encoder_attention_heads = 8 #was 8
args.attention_dropout = 0.1
args.topk_ratio = 1.0
args.dropout = 0.2
args.encoder_normalize_before = True
args.encoder_ffn_embed_dim = 2048
args.use_nfm = 'false'
args.shared_memory_attention = False
args.self_attention = True
args.mem_slots = 4
args.use_topk = False
args.topk = 8
args.num_steps = 5

from .transformer_utilities.transformer_layer import TransformerEncoderLayerVanilla
from .transformer_utilities.pos_enc import PositionEncoder
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder source from "Coordination Among Neural Modules Through a Shared Global Workspace"
    """
    def __init__(self,
                 embed_dim,
                 ffn_dim,
                 num_layers = 6,
                 num_heads = 4,
                 dropout = 0.1,
                 shared_memory_attention = False,
                 shared_memory_percentage = 0.1,
                 share_parameters = False,
                 mem_slots = 8,
                 num_attention_schemas = 3,
                 num_gru_schemas = 3,
                 use_topk = False,
                 topk = 3,
                 num_steps = 5,
                 null_attention = False,
                 regressive = False,
                 mode = 'full'):
        super().__init__()

        args.mem_slots = mem_slots
        args.encoder_embed_dim = embed_dim
        args.encoder_ffn_embed_dim = ffn_dim
        args.encoder_attention_heads = num_heads
        args.dropout = dropout
        args.shared_memory_attention = shared_memory_attention
        args.num_steps = num_steps
        args.null_attention = null_attention
        args.regressive = regressive

        self.num_layers = num_layers
        self.shared_memory_attention = shared_memory_attention
        self.shared_memory_percentage = shared_memory_percentage
        self.mode = mode

        layer_lst = []
        args.use_topk = use_topk
        args.topk = topk

        args.encoder_embed_dim = embed_dim
        self.share_parameters = share_parameters
        if share_parameters:
            self.enc = TransformerEncoderLayerVanilla(args)
        else:
            layer_lst = []
            for i in range(self.num_layers):
                layer_lst.append(TransformerEncoderLayerVanilla(args))
            self.layers = nn.ModuleList(layer_lst)

        self.pe = PositionEncoder(args.encoder_embed_dim, max_seq_len=args.num_steps)

    def forward(self, x, mask = None, num_layers = None):

        x = x.permute(1, 0, 2)

        # Apply positional encoding only when mode is 'full'
        if self.mode == 'full':
            x = self.pe(x)

        if self.shared_memory_attention:
            memory_size = int(self.shared_memory_percentage * x.size(0))

            memory = torch.randn(memory_size, 1, x.size(2)).repeat(1 ,x.size(1), 1).to(x.device)
        else:
            memory = None
        if self.shared_memory_attention:
            if self.share_parameters:
                if self.enc.self_attn.memory is not None:
                    self.enc.self_attn.init_memory(x.size(1), x.size(0), x.device)
            else:
                for layer in self.layers:
                    if layer.self_attn.memory is not None:
                        layer.self_attn.init_memory(x.size(1), x.device)

        
        for i in range(self.num_layers):
            if self.share_parameters:
                x, memory, attention_x = self.enc(x, mask, memory = memory)
            else:
                x, memory, attention_x = self.layers[i](x, mask, memory = memory)
        return x.permute(1, 0, 2), memory, attention_x

