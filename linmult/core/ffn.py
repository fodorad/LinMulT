"""FFN residual block: two linear layers with GELU activation and residual connection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNResidual(nn.Module):
    """Two-layer FFN with GELU activation, dropout, and residual connection.

    Computes ``x + fc2(dropout(gelu(fc1(x))))``.

    Args:
        dim (int): Input and output feature dimension.
        dropout (float): Dropout probability applied after the first linear layer.
            Defaults to ``0.0``.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN + residual.

        Args:
            x (torch.Tensor): Input tensor of any shape with last dim ``dim``.

        Returns:
            torch.Tensor: Same shape as ``x``.
        """
        return self.fc2(F.dropout(F.gelu(self.fc1(x)), p=self.dropout, training=self.training)) + x
