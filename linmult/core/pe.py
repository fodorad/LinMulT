"""Sinusoidal positional encoding with optional dropout."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence inputs.

    Adds fixed sinusoidal position encodings to the input tensor, following
    Vaswani et al. (2017). The encoding matrix is computed lazily and cached;
    it is only recomputed when the sequence is longer or the feature dimension
    changes.

    Args:
        dropout (float): Dropout probability applied after adding the encoding.
            Defaults to ``0.1``.
    """

    def __init__(self, dropout: float = 0.1):
        """Initialize PositionalEncoding."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to the input.

        The encoding matrix is rebuilt only when the cached tensor is shorter
        than the current sequence or the feature dimensionality has changed.
        For odd feature dimensions, the cosine slot count is ``floor(F/2)``
        while the sine slot count is ``ceil(F/2)``; the division term is
        sliced accordingly so no index is out of range.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.

        Returns:
            torch.Tensor: Encoded tensor of shape ``(B, T, F)`` with dropout applied.
        """
        _, time_dim, feature_dim = x.shape

        # Rebuild only when the cached PE is too short or the feature dim changed.
        # A larger cache is reused by slicing, avoiding recomputation on shorter sequences.
        if self.pe is None or self.pe.size(1) < time_dim or self.pe.size(2) != feature_dim:
            pe = torch.zeros(time_dim, feature_dim, device=x.device)
            position = torch.arange(0, time_dim, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, feature_dim, 2, dtype=torch.float, device=x.device)
                * (-math.log(10000.0) / feature_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            # For odd feature_dim, 1::2 has floor(F/2) slots but div_term has ceil(F/2) — slice.
            pe[:, 1::2] = torch.cos(position * div_term[: feature_dim // 2])
            self.pe = pe.unsqueeze(0)  # (1, T, F)

        x = x + self.pe[:, :time_dim, :]
        return self.dropout(x)
