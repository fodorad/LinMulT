"""Temporal Convolutional Network (TCN) for local temporal smoothing.

Provides dilated causal 1-D convolution layers that capture short-range
temporal dynamics (e.g. micro-expressions, motion patterns) without leaking
future information.  Designed to sit after the projection Conv1d and before
cross-modal attention in the LinMulT pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNLayer(nn.Module):
    """Single dilated causal Conv1d layer with residual connection.

    Computes::

        x + dropout(relu(bn(causal_conv1d(x))))

    Causal padding is applied on the left so that output at time *t* depends
    only on inputs at times ``<= t``.

    Args:
        d_model (int): Number of input and output channels.
        kernel_size (int): Convolution kernel size.  Defaults to ``3``.
        dilation (int): Dilation factor.  Defaults to ``1``.
        dropout (float): Dropout probability after activation.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution with residual.

        Args:
            x (torch.Tensor): Input ``(B, T, d_model)``.

        Returns:
            torch.Tensor: Output ``(B, T, d_model)``, same shape as input.
        """
        residual = x
        # (B, T, C) -> (B, C, T)
        out = x.transpose(1, 2)
        out = F.pad(out, (self.left_pad, 0))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # (B, C, T) -> (B, T, C)
        out = out.transpose(1, 2)
        return out + residual


class TCN(nn.Module):
    """Stack of :class:`TCNLayer` with exponentially increasing dilation.

    Dilations are ``[1, 2, 4, ..., 2^(num_layers-1)]``, giving a receptive
    field of ``1 + sum((kernel_size - 1) * 2^i for i in range(num_layers))``
    frames.  With the defaults (``num_layers=3, kernel_size=3``) the receptive
    field is 15 frames (~0.5 s at 30 fps).

    Args:
        d_model (int): Channel dimension (preserved through all layers).
        num_layers (int): Number of dilated convolution layers.  Defaults to ``3``.
        kernel_size (int): Kernel size for every layer.  Defaults to ``3``.
        dropout (float): Dropout probability in each layer.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TCNLayer(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all TCN layers sequentially.

        Args:
            x (torch.Tensor): Input ``(B, T, d_model)``.

        Returns:
            torch.Tensor: Output ``(B, T, d_model)``, temporally smoothed.
        """
        for layer in self.layers:
            x = layer(x)
        return x
