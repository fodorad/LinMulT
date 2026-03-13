"""Transformer encoder: stacked pre-norm layers with multi-head attention and FFN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from linmult.core.attention import AttentionConfig, AttentionFactory
from linmult.core.pe import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple stacked layers.

    Supports both self-attention (when ``x_k`` and ``x_v`` are omitted) and
    cross-modal attention (when ``x_k`` and ``x_v`` are provided).

    Args:
        d_model (int): Input and output feature dimensionality. Defaults to ``40``.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        num_layers (int): Number of stacked encoder layers. Defaults to ``6``.
        attention_config (AttentionConfig, optional): Attention type and parameters.
            Defaults to ``AttentionConfig()`` (linear attention).
        dropout_pe (float): Dropout after positional encoding. Defaults to ``0.0``.
        dropout_ffn (float): Dropout in the FFN sub-layer. Defaults to ``0.1``.
        is_cross_modal (bool): Allocate a separate layer-norm for cross-modal key input.
            Set to ``True`` for cross-modal attention encoders. Defaults to ``False``.
        name (str): Module name shown in ``repr``. Defaults to ``""``.
    """

    def __init__(
        self,
        d_model: int = 40,
        num_heads: int = 8,
        num_layers: int = 6,
        attention_config: AttentionConfig | None = None,
        dropout_pe: float = 0.0,
        dropout_ffn: float = 0.1,
        is_cross_modal: bool = False,
        name: str = "",
    ):
        super().__init__()

        self.name = name
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = PositionalEncoding(dropout=dropout_pe)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    attention_config=attention_config,
                    dropout=dropout_ffn,
                    is_cross_modal=is_cross_modal,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def extra_repr(self) -> str:
        """Return the module name for identification in repr output."""
        return f"name={self.name!r}"  # pragma: no cover

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the transformer encoder.

        When ``x_k`` and ``x_v`` are omitted the encoder runs self-attention
        (``x_q == x_k == x_v``). When provided it runs cross-modal attention.

        Args:
            x_q (torch.Tensor): Query input of shape ``(B, T_1, F)``.
            x_k (torch.Tensor, optional): Key input of shape ``(B, T_2, F)``.
            x_v (torch.Tensor, optional): Value input of shape ``(B, T_2, F)``.
            query_mask (torch.BoolTensor, optional): Mask for queries, shape ``(B, T_1)``.
            key_mask (torch.BoolTensor, optional): Mask for keys, shape ``(B, T_2)``.

        Returns:
            torch.Tensor: Encoded output of shape ``(B, T_1, F)``.
        """
        x = self.embed_positions(self.embed_scale * x_q)  # (B, T, d_model)

        if x_k is not None and x_v is not None:
            # K and V are always the same source tensor; apply PE once, reuse for both.
            x_k = x_v = self.embed_positions(self.embed_scale * x_k)

            for layer in self.layers:
                x = layer(x, x_k, x_v, query_mask=query_mask, key_mask=key_mask)
        else:
            for layer in self.layers:
                x = layer(x, query_mask=query_mask)

        return self.layer_norm(x)


class TransformerEncoderLayer(nn.Module):
    """Single pre-norm transformer encoder layer with attention + FFN.

    Supports self-attention and cross-modal attention. The cross-modal layer
    norm (``layer_norm_cross``) is only allocated when ``cross_modal=True``,
    since pure self-attention layers never receive external keys.

    Args:
        d_model (int): Feature dimensionality. Defaults to ``40``.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        attention_config (AttentionConfig, optional): Attention type and parameters.
            Defaults to ``AttentionConfig()`` (linear attention).
        dropout (float): Dropout on FFN and residual connections. Defaults to ``0.1``.
        is_cross_modal (bool): Allocate a cross-modal layer-norm. Defaults to ``False``.
    """

    def __init__(
        self,
        d_model: int = 40,
        num_heads: int = 8,
        attention_config: AttentionConfig | None = None,
        dropout: float = 0.1,
        is_cross_modal: bool = False,
    ):
        super().__init__()

        self.attention_type = attention_config.type if attention_config is not None else "linear"
        self.attention = AttentionFactory.create(d_model, num_heads, attention_config)

        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        # Allocate cross-modal norm only for layers that actually do cross-modal attention;
        # pure self-attention layers (SAT) never receive x_k/x_v.
        self.is_cross_modal = is_cross_modal
        if self.is_cross_modal:
            self.layer_norm_cross = nn.LayerNorm(d_model)

        self.dropout = dropout

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one transformer encoder layer.

        Args:
            x_q (torch.Tensor): Query input of shape ``(B, T_1, F)``.
            x_k (torch.Tensor, optional): Key input of shape ``(B, T_2, F)``.
            x_v (torch.Tensor, optional): Value input of shape ``(B, T_2, F)``.
            query_mask (torch.BoolTensor, optional): Mask for queries, shape ``(B, T_1)``.
            key_mask (torch.BoolTensor, optional): Mask for keys, shape ``(B, T_2)``.

        Returns:
            torch.Tensor: Layer output of shape ``(B, T_1, F)``.

        Raises:
            ValueError: If mask shapes or dtypes are incorrect.
        """
        if query_mask is not None and (
            query_mask.shape != x_q.shape[:2] or query_mask.dtype != torch.bool
        ):
            raise ValueError(
                f"Expected query mask has shape (B, T_1) and bool dtype, "
                f"got instead: {query_mask.shape} and {query_mask.dtype}"
            )

        if key_mask is not None:
            if x_k is None:
                raise ValueError("key_mask was provided but x_k is None.")
            if key_mask.shape != x_k.shape[:2] or key_mask.dtype != torch.bool:
                raise ValueError(
                    f"Expected key_mask of shape (B, T_2) and bool dtype, "
                    f"got {key_mask.shape} {key_mask.dtype}."
                )

        residual = x_q
        x_q = self.layer_norms[0](x_q)

        if x_k is not None and x_v is not None:
            # K and V are always the same source tensor — normalize once, reuse for both.
            # Apply layer_norm_cross only when allocated (cross_modal=True layers).
            # Non-cross-modal encoders receive already-normed features from their own encoder.
            x_kv = self.layer_norm_cross(x_k) if self.is_cross_modal else x_k  # (B, T_2, F)

            if self.attention_type == "mha":
                kpm = ~key_mask if key_mask is not None else None
                x_q, _ = self.attention(x_q, x_kv, x_kv, key_padding_mask=kpm)
            else:
                x_q, _ = self.attention(
                    x_q, x_kv, x_kv, query_mask=query_mask, key_mask=key_mask
                )  # (B, T_1, F)

            if key_mask is not None:
                fully_masked_keys = ~key_mask.any(dim=1)  # (B,)
                if fully_masked_keys.any():
                    # Samples where all keys are masked should produce zero output.
                    zero_output = torch.zeros_like(x_q)
                    x_q = torch.where(fully_masked_keys.unsqueeze(1).unsqueeze(2), zero_output, x_q)

        else:  # self-attention
            if self.attention_type == "mha":
                kpm = ~query_mask if query_mask is not None else None
                x_q, _ = self.attention(x_q, x_q, x_q, key_padding_mask=kpm)
            else:
                x_q, _ = self.attention(
                    x_q, x_q, x_q, query_mask=query_mask, key_mask=query_mask
                )  # (B, T_1, F)

        if query_mask is not None:
            # Zero out ALL padded query positions (not just fully-masked batches).
            # Softmax/BigBird produce NaN at positions where the combined attn_mask row is
            # all -inf (query_mask[b,i]=False → combined_mask row all-False → softmax → NaN).
            # Must use masked_fill (not multiplication): NaN * 0 = NaN in IEEE 754, so a simple
            # x * mask would still propagate NaN. masked_fill unconditionally writes 0.0.
            x_q = x_q.masked_fill(~query_mask.unsqueeze(-1), 0.0)

        x_q = F.dropout(x_q, p=self.dropout, training=self.training)
        x_q = residual + x_q

        residual = x_q
        x_q = self.layer_norms[1](x_q)

        x_q = F.gelu(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.dropout, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.dropout, training=self.training)
        x_q = residual + x_q

        return x_q
