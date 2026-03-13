"""Temporal reducers, aligners, and composite modules (TRM, TAM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from linmult.core.attention import AttentionConfig
from linmult.core.transformer import TransformerEncoder


class TemporalFactory:
    """Factory for creating temporal signal aligners and reducers."""

    @staticmethod
    def create_aligner(method: str = "aap") -> nn.Module:
        """Create a temporal aligner module.

        Args:
            method (str): Aligner type. One of:

                - ``"aap"``: Adaptive average pooling.
                - ``"amp"``: Adaptive max pooling.
                - ``"padding"``: Zero-padding / truncation.

        Returns:
            nn.Module: The constructed aligner module.

        Raises:
            ValueError: If ``method`` is not one of the supported values.
        """
        registry = {
            "aap": AdaptiveAvgPooling,
            "amp": AdaptiveMaxPooling,
            "padding": TemporalPadding,
        }
        if method not in registry:  # pragma: no cover
            raise ValueError(f"Unknown aligner: {method!r}")
        return registry[method]()

    @staticmethod
    def create_reducer(d_model: int, reducer: str) -> nn.Module:
        """Create a temporal reducer module.

        Args:
            d_model (int): Feature dimensionality of the input tensor.
                Required for :class:`AttentionPooling`; ignored by other reducers.
            reducer (str): Reducer type. One of ``"attentionpool"``,
                ``"gmp"``, ``"gap"``, ``"last"``.

        Returns:
            nn.Module: The constructed reducer module.

        Raises:
            ValueError: If ``reducer`` is not one of the supported values.
        """
        if reducer == "attentionpool":
            return AttentionPooling(d_model)

        registry = {
            "gmp": GlobalMaxPooling,
            "gap": GlobalAvgPooling,
            "last": LastTimestamp,
        }
        if reducer not in registry:  # pragma: no cover
            raise ValueError(f"Unknown reducer: {reducer!r}")
        return registry[reducer]()


class TemporalPadding(nn.Module):
    """Temporal aligner via truncation or zero-padding.

    Adjusts the time dimension of a tensor to exactly ``time_dim`` by
    truncating if the sequence is too long, or zero-padding if too short.
    The mask is updated accordingly (padded positions are ``False``).
    """

    def forward(
        self, x: torch.Tensor, time_dim: int, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate or pad the time dimension.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            time_dim (int): Target time dimension.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid. If ``None``, all input positions are treated as valid.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor of shape
                ``(B, time_dim, F)`` and updated mask of shape ``(B, time_dim)``.
        """
        current_time_dim = x.size(1)

        if current_time_dim > time_dim:
            x_new = x[:, :time_dim, :]
            if mask is not None:
                mask_new = mask[:, :time_dim]
            else:
                mask_new = torch.ones((x.size(0), time_dim), dtype=torch.bool, device=x.device)

        elif current_time_dim < time_dim:
            pad_size = time_dim - current_time_dim
            x_new = F.pad(x, (0, 0, 0, pad_size))  # pad along the time dimension

            if mask is not None:
                mask_new = F.pad(mask, (0, pad_size), value=False)
            else:
                mask_new = torch.ones(
                    (x.size(0), current_time_dim), dtype=torch.bool, device=x.device
                )
                mask_new = F.pad(mask_new, (0, pad_size), value=False)

        else:
            x_new = x
            mask_new = (
                mask
                if mask is not None
                else torch.ones((x.size(0), current_time_dim), dtype=torch.bool, device=x.device)
            )

        return x_new, mask_new


class AdaptiveMaxPooling(nn.Module):
    """Temporal aligner via adaptive max pooling.

    Resizes the time dimension of a tensor to ``time_dim`` using
    ``F.adaptive_max_pool1d``. Masked (padded) positions are filled with
    ``-inf`` before pooling so they never win the max, and the output mask
    is derived from the result.
    """

    def forward(
        self, x: torch.Tensor, time_dim: int, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive max pooling along the time dimension.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            time_dim (int): Target time dimension.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pooled tensor of shape
                ``(B, time_dim, F)`` and updated mask of shape ``(B, time_dim)``.
        """
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand_as(x)  # (B, T, F)
            x = x.masked_fill(~expanded_mask, float("-inf"))

        x_new = F.adaptive_max_pool1d(x.transpose(1, 2), time_dim).transpose(1, 2)

        if mask is not None:
            mask_new = (x_new != float("-inf")).any(dim=-1)  # (B, time_dim)
            # masked_fill (non-in-place) avoids breaking the autograd graph
            x_new = x_new.masked_fill(x_new == float("-inf"), 0.0)
        else:
            mask_new = torch.ones(
                x_new.size(0), x_new.size(1), dtype=torch.bool, device=x_new.device
            )

        return x_new, mask_new


class AdaptiveAvgPooling(nn.Module):
    """Temporal aligner via adaptive average pooling.

    Resizes the time dimension of a tensor to ``time_dim`` using
    ``F.adaptive_avg_pool1d``. Masked positions contribute zero to the average
    and the output is renormalized by the fraction of valid input positions in
    each output bin.
    """

    def forward(
        self, x: torch.Tensor, time_dim: int, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive average pooling along the time dimension.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            time_dim (int): Target time dimension.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pooled tensor of shape
                ``(B, time_dim, F)`` and updated mask of shape ``(B, time_dim)``.
        """
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).expand_as(x)  # (B, T, F)
            x = x.masked_fill(~expanded_mask, 0.0)

            pooled_mask = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), time_dim).squeeze(
                1
            )  # (B, time_dim)
            mask_new = pooled_mask > 1e-8

            x_pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), time_dim).transpose(
                1, 2
            )  # (B, time_dim, F)

            # Renormalize to account for masked regions.
            # clamp(min=1e-8) avoids division by zero for fully-masked bins.
            x_new = x_pooled / pooled_mask.unsqueeze(-1).clamp(min=1e-8)  # (B, time_dim, F)
        else:
            x_new = F.adaptive_avg_pool1d(x.transpose(1, 2), time_dim).transpose(
                1, 2
            )  # (B, time_dim, F)
            mask_new = torch.ones(x.size(0), time_dim, dtype=torch.bool, device=x.device)

        return x_new, mask_new


class LastTimestamp(nn.Module):
    """Temporal reducer that extracts the last valid timestamp.

    If a mask is provided, selects the feature vector at the last ``True``
    position for each sample. Fully-masked samples (all ``False``) return
    a zero vector.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Extract the last valid timestamp.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                If ``None``, the final timestep is selected for all samples.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, F)``.
        """
        if mask is not None:
            valid_counts = mask.sum(dim=1).long()  # (B,)
            last_timestamp_index = (valid_counts - 1).clamp(min=0)
        else:
            last_timestamp_index = torch.full(
                (x.size(0),), x.size(1) - 1, dtype=torch.long, device=x.device
            )

        batch_indices = torch.arange(x.size(0), device=x.device)
        result = x[batch_indices, last_timestamp_index]  # (B, F)

        if mask is not None:
            fully_masked = valid_counts == 0  # (B,)
            if fully_masked.any():
                result = result * (~fully_masked).float().unsqueeze(-1)

        return result  # (B, F)


class GlobalAvgPooling(nn.Module):
    """Temporal reducer via masked global average pooling.

    Computes the mean over valid (unmasked) timesteps. If no mask is provided,
    averages over all timesteps.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply global average pooling.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid. If ``None``, all positions are treated as valid.

        Returns:
            torch.Tensor: Pooled output of shape ``(B, F)``.
        """
        if mask is not None:
            expanded_mask = mask.unsqueeze(-1)  # (B, T, 1)
            x = x * expanded_mask
            sum_x = x.sum(dim=1)  # (B, F)
            count_x = expanded_mask.sum(dim=1).clamp(min=1)  # (B, 1)
            return sum_x / count_x  # (B, F)
        else:
            return x.mean(dim=1)  # (B, F)


class GlobalMaxPooling(nn.Module):
    """Temporal reducer via masked global max pooling.

    Computes the max over valid (unmasked) timesteps. Masked positions are
    filled with ``-inf`` before the max, and fully-masked samples return zero.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply global max pooling.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid. If ``None``, all positions are treated as valid.

        Returns:
            torch.Tensor: Pooled output of shape ``(B, F)``.
        """
        fully_masked = None
        if mask is not None:
            fully_masked = ~mask.any(dim=1)  # (B,) — fully masked samples
            x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        result = x.max(dim=1)[0]  # (B, F)
        if fully_masked is not None and fully_masked.any():
            result = result.masked_fill(fully_masked.unsqueeze(-1), 0.0)
        return result  # (B, F)


class AttentionPooling(nn.Module):
    """Temporal reducer via learned attention-weighted pooling.

    Learns a scalar attention score per timestep and computes a weighted sum
    of the input features. Masked positions receive ``-inf`` before the softmax
    so their weight is zero. Fully-masked samples return a zero vector.

    Args:
        d_model (int): Input feature dimensionality.
    """

    def __init__(self, d_model: int):
        """Initialize AttentionPooling."""
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention-weighted pooling.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, T, F)``.
            mask (torch.BoolTensor, optional): Validity mask of shape ``(B, T)``.
                True = valid.

        Returns:
            torch.Tensor: Pooled output of shape ``(B, F)``.
        """
        attn_weights = self.attention(x).squeeze(-1)  # (B, T)
        all_masked = None
        if mask is not None:
            all_masked = (~mask).all(dim=1)  # (B,) — fully masked samples
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
            if all_masked.any():
                # Replace -inf with 0 so softmax gives uniform weights instead of NaN
                safe = torch.zeros_like(attn_weights)
                attn_weights = torch.where(all_masked.unsqueeze(1), safe, attn_weights)

        attn_weights = torch.softmax(attn_weights, dim=-1)  # (B, T)
        weighted_avg = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, F)

        # Zero out fully-masked samples (uniform weights would give non-zero average)
        if all_masked is not None and all_masked.any():
            weighted_avg = weighted_avg * (~all_masked).float().unsqueeze(-1)

        return weighted_avg  # (B, F)


class TRM(nn.Module):
    """Time Reduce Module: aggregates the time dimension of a sequence tensor.

    Transforms ``(B, T, F)`` → ``(B, F)`` using a configurable pooling strategy.

    Args:
        d_model (int): Input feature dimensionality. Required for ``"attentionpool"``;
            ignored by ``"gap"``, ``"gmp"``, and ``"last"``.
        reducer (str): Pooling strategy. One of ``"attentionpool"``, ``"gmp"``,
            ``"gap"``, ``"last"``.
    """

    def __init__(self, d_model: int, reducer: str):
        """Initialize TRM."""
        super().__init__()

        if reducer not in {"attentionpool", "gmp", "gap", "last"}:
            raise ValueError(f"Invalid reducer: {reducer!r}")

        self.time_dim_reducer = TemporalFactory.create_reducer(d_model=d_model, reducer=reducer)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reduce the time dimension.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, F)``.
            mask (torch.Tensor, optional): Validity mask of shape ``(B, T)``. True = valid.

        Returns:
            torch.Tensor: Reduced output of shape ``(B, F)``.
        """
        return self.time_dim_reducer(x, mask)  # (B, T, F) -> (B, F)

    def apply_to_list(
        self, x_list: list[torch.Tensor], mask_list: list[torch.Tensor | None]
    ) -> list[torch.Tensor]:
        """Apply time reduction independently to each tensor in a list.

        Args:
            x_list (list[torch.Tensor]): List of tensors, each of shape ``(B, T, F)``.
            mask_list (list[torch.Tensor | None]): Corresponding masks, each of shape ``(B, T)``.

        Returns:
            list[torch.Tensor]: List of reduced tensors, each of shape ``(B, F)``.
        """
        return [self(x, mask) for x, mask in zip(x_list, mask_list)]


class TAM(nn.Module):
    """Time Align Module: aligns the time dimensions of multiple tensors.

    Transforms a list of ``(B, T_i, F)`` tensors to a single fused tensor
    ``(B, time_dim, tgt_dim)`` by pooling/padding each sequence to a common
    ``time_dim``, concatenating along the feature axis, processing with a
    transformer, and projecting to ``tgt_dim``.

    Args:
        input_dim (int): Concatenated input dimensionality (sum of feature dims across modalities).
        output_dim (int): Output feature dimensionality after projection.
        aligner (str): Temporal alignment strategy. One of ``"aap"``, ``"amp"``,
            ``"padding"``.
        time_dim (int): Target time dimension after alignment.
        dropout_out (float): Dropout in the output projector. Defaults to ``0.1``.
        num_layers (int): Depth of the internal transformer encoder. Defaults to ``6``.
        num_heads (int): Number of attention heads in the internal encoder. Defaults to ``8``.
        attention_config (AttentionConfig, optional): Attention type and parameters for the
            internal encoder. Defaults to ``AttentionConfig()`` (linear attention).
        dropout_pe (float): Positional-encoding dropout for the internal encoder.
            Defaults to ``0.0``.
        dropout_ffn (float): FFN dropout for the internal encoder. Defaults to ``0.1``.
        name (str): Module name shown in ``repr``. Defaults to ``""``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        aligner: str,
        time_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        attention_config: AttentionConfig | None = None,
        dropout_pe: float = 0.0,
        dropout_ffn: float = 0.1,
        dropout_out: float = 0.1,
        name: str = "",
    ):
        """Initialize TAM."""
        super().__init__()

        if aligner not in {"aap", "amp", "padding"}:
            raise ValueError(f"Invalid aligner: {aligner!r}")

        self.name = name
        self.aligned_time_dim = time_dim
        self.time_dim_aligner = TemporalFactory.create_aligner(aligner)
        self.transformer = TransformerEncoder(
            d_model=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attention_config=attention_config,
            dropout_pe=dropout_pe,
            dropout_ffn=dropout_ffn,
        )
        self.projector = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout_out),
            nn.Conv1d(output_dim, output_dim, kernel_size=1, padding=0, bias=True),
        )

    def extra_repr(self) -> str:
        """Return the module name for identification in repr output."""
        return f"name={self.name!r}"  # pragma: no cover

    def forward(
        self, x_list: list[torch.Tensor], mask_list: list[torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align, fuse, and project multiple sequences.

        Args:
            x_list (list[torch.Tensor]): Input tensors, each of shape ``(B, T_i, F)``.
            mask_list (list[torch.BoolTensor | None]): Corresponding masks, each of
                shape ``(B, T_i)`` or ``None`` (treated as all-valid).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Aligned tensor of shape
                ``(B, time_dim, output_dim)`` and validity mask of shape ``(B, time_dim)``.
        """
        # Align each input to time_dim
        aligned = [
            self.time_dim_aligner(x, self.aligned_time_dim, mask)
            for x, mask in zip(x_list, mask_list)
        ]
        aligned_x = [a[0] for a in aligned]
        aligned_masks = [a[1] for a in aligned]

        # AND across modalities to get a joint mask
        joint_mask = torch.stack(aligned_masks, dim=0).all(dim=0)  # (B, time_dim)

        # Concatenate along the feature axis and run the transformer
        x = torch.cat(aligned_x, dim=-1)  # (B, time_dim, src_dim)
        x = self.transformer(x, query_mask=joint_mask)  # (B, time_dim, src_dim)

        # Project to tgt_dim
        x = self.projector(x.transpose(1, 2)).transpose(1, 2)  # (B, time_dim, tgt_dim)

        return x, joint_mask
