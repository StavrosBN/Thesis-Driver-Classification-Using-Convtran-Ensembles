"""
Customized from the original repository for ConvTran (Foumani et al., 2023).
https://github.com/Navidfoumani/ConvTran
tAPE (time Absolute Position Encoding)
"""
import math
import torch
import torch.nn as nn


class tAPE(nn.Module):
    """
    The key difference from the classic sinusoidal position encoding (Vaswani et al.)
    is the term (d_model / max_len) it incorporates the
    sequence length (max_len) into the frequency, ensuring the encoding works
    correctly even with low embedding dimensions (which is common in time series,
    unlike in NLP, where d_model is typically 512+)
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe
        return self.dropout(x)
    