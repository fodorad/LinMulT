import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', None, persistent=False)

    def forward(self, x):
        # x shape: (B, T, F)
        batch_size, time_dim, feature_dim = x.shape
        if self.pe is None or self.pe.size(1) < time_dim or self.pe.size(2) != feature_dim:
            pe = torch.zeros(time_dim, feature_dim, device=x.device)
            position = torch.arange(0, time_dim, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, feature_dim, 2, dtype=torch.float, device=x.device) * (-math.log(10000.0) / feature_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0) # pe shape: (1, T, F)
            self.pe = pe

        x = x + self.pe[:, :time_dim, :].clone().detach()
        x = self.dropout(x)
        return x