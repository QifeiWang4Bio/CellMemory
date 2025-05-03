import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import random

class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.pos_emb_weight = nn.Parameter(torch.ones_like(pe))

    def forward(self, x):
        # make embeddings relatively larger
        x = x.permute(1,0,2)

        seq_len = x.size(1)

        pe_use = Variable(self.pe[:,:seq_len] * F.sigmoid(self.pos_emb_weight[:,:seq_len]), requires_grad=False).cuda()

        x = x + pe_use

        x = x.permute(1,0,2)

        return x
