"""Batch normalization and instance normalization wrappers."""

import torch
import torch.nn as nn


class BN(nn.Module):
    """Batch normalization for both sequence ``(B, T, F)`` and vector ``(B, F)`` inputs.

    Args:
        feature_dim (int): Number of features to normalize.
        time_aware (bool): If ``True``, expects sequence inputs ``(B, T, F)`` and
            normalizes over the batch and time axes via ``BatchNorm1d``.
            If ``False``, expects vector inputs ``(B, F)``.
    """

    def __init__(self, feature_dim: int, time_aware: bool):
        """Initialize BN."""
        super().__init__()
        self.time_aware = time_aware
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization.

        Args:
            x (torch.Tensor): Shape ``(B, T, F)`` when ``time_aware=True``,
                or ``(B, F)`` otherwise.

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        if self.time_aware and x.ndim == 3:  # (B, T, F)
            x = x.permute(0, 2, 1)  # (B, F, T)
            x = self.bn(x)
            return x.permute(0, 2, 1)  # (B, T, F)
        return self.bn(x)  # (B, F)


class IN(nn.Module):
    """Instance/layer normalization for sequence ``(B, T, F)`` and vector ``(B, F)`` inputs.

    For sequences (``time_aware=True``): applies ``InstanceNorm1d``, normalizing
    each sample and channel independently over the time axis.

    For vectors (``time_aware=False``): applies ``LayerNorm`` over the feature
    axis per sample. ``InstanceNorm1d`` is degenerate for single-element sequences
    (normalizes a scalar to zero), so ``LayerNorm`` is the correct choice here.

    Args:
        feature_dim (int): Number of features to normalize.
        time_aware (bool): Determines the normalization strategy (see above).
    """

    def __init__(self, feature_dim: int, time_aware: bool):
        """Initialize IN."""
        super().__init__()
        self.time_aware = time_aware
        if time_aware:
            self.in_norm = nn.InstanceNorm1d(feature_dim, affine=True)
        else:
            # For vector inputs (B, F): LayerNorm normalizes over features per sample,
            # which is the correct semantics (InstanceNorm1d with L=1 is degenerate).
            self.in_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply instance or layer normalization.

        Args:
            x (torch.Tensor): Shape ``(B, T, F)`` when ``time_aware=True``,
                or ``(B, F)`` otherwise.

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        if self.time_aware and x.ndim == 3:  # (B, T, F)
            x = x.permute(0, 2, 1)  # (B, F, T)
            x = self.in_norm(x)
            return x.permute(0, 2, 1)  # (B, T, F)
        return self.in_norm(x)  # (B, F)
