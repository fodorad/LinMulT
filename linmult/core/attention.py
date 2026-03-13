"""Attention mechanisms: linear, softmax, BigBird, Performer (FAVOR+), and GAU (flash)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    """Attention mechanism selection and its type-specific hyperparameters.

    This is an internal construction spec — created from user-facing config fields
    in ``LinT`` or ``LinMulT``, then passed directly to :class:`TransformerEncoder`
    and :class:`TAM`. Each field is only relevant when ``type`` matches.

    Args:
        type (str): Attention mechanism. One of ``"linear"`` (default),
            ``"performer"``, ``"flash"``, ``"softmax"``, ``"bigbird"``, ``"mha"``.
        dropout (float): Dropout probability on attention weights. Defaults to ``0.0``.
        flash_query_key_dim (int | None): Scoring dimension for ``"flash"`` (GAU).
            Defaults to ``None`` (computed as ``max(d_model // 2, 16)``).
        performer_num_random_features (int | None): Random feature count for ``"performer"``.
            Defaults to ``None`` (computed as ``max(head_dim * 4, 32)``).
        bigbird_block_size (int): Local block size for ``"bigbird"``. Defaults to ``64``.
        bigbird_num_global_tokens (int): Global tokens for ``"bigbird"``. Defaults to ``16``.
        bigbird_num_random_tokens (int): Random tokens for ``"bigbird"``. Defaults to ``10``.
    """

    type: str = "linear"
    dropout: float = 0.0
    flash_query_key_dim: int | None = None  # Flash only
    performer_num_random_features: int | None = None  # Performer only
    bigbird_block_size: int = 64  # BigBird only
    bigbird_num_global_tokens: int = 16  # BigBird only
    bigbird_num_random_tokens: int = 10  # BigBird only


class AttentionFactory:
    """Factory for creating attention layers from an :class:`AttentionConfig`."""

    @staticmethod
    def create(
        d_model: int, num_heads: int, attention_config: AttentionConfig | None = None
    ) -> nn.Module:
        """Create and return an attention layer.

        Args:
            d_model (int): Input feature dimensionality.
            num_heads (int): Number of attention heads.
            attention_config (AttentionConfig, optional): Attention configuration.
                Defaults to ``AttentionConfig()`` (linear attention).

        Returns:
            nn.Module: An attention module. ``"mha"`` returns
                ``nn.MultiheadAttention``; ``"flash"`` (GAU) returns
                :class:`GatedAttentionUnit`; all others return :class:`AttentionLayer`.

        Raises:
            ValueError: If ``attention_config.type`` is not one of the supported values.
        """
        if attention_config is None:
            attention_config = AttentionConfig()

        attention_type = attention_config.type

        if attention_type not in {"linear", "performer", "softmax", "bigbird", "mha", "flash"}:
            raise ValueError(
                f"Given attention type ({attention_type!r}) is not supported. "
                "Choose from {'bigbird', 'flash', 'linear', 'performer', 'softmax', 'mha'}."
            )

        if attention_type == "bigbird":
            return AttentionLayer(
                BigBirdAttention(
                    num_heads=num_heads,
                    block_size=attention_config.bigbird_block_size,
                    num_global_tokens=attention_config.bigbird_num_global_tokens,
                    num_random_tokens=attention_config.bigbird_num_random_tokens,
                    dropout=attention_config.dropout,
                ),
                d_model=d_model,
                num_heads=num_heads,
            )
        elif attention_type == "linear":
            return AttentionLayer(
                LinearAttention(d_model=d_model, num_heads=num_heads),
                d_model=d_model,
                num_heads=num_heads,
            )
        elif attention_type == "performer":
            return AttentionLayer(
                LinearAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    feature_map=PerformerFeatureMap.factory(
                        num_features=attention_config.performer_num_random_features
                    ),
                ),
                d_model=d_model,
                num_heads=num_heads,
            )
        elif attention_type == "softmax":
            return AttentionLayer(
                SoftmaxAttention(
                    d_model=d_model, num_heads=num_heads, dropout=attention_config.dropout
                ),
                d_model=d_model,
                num_heads=num_heads,
            )
        elif attention_type == "flash":
            return GatedAttentionUnit(
                d_model=d_model,
                query_key_dim=attention_config.flash_query_key_dim,
                dropout=attention_config.dropout,
            )
        else:  # mha
            return nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=attention_config.dropout,
                batch_first=True,
            )


class AttentionLayer(nn.Module):
    """Multi-head attention wrapper that projects inputs and reprojects the output.

    Projects queries, keys, and values to multi-head representations, delegates
    the actual attention computation to an inner attention module, then reprojects
    the concatenated heads back to ``d_model``.

    Args:
        attention (nn.Module): Inner attention module (e.g. ``LinearAttention``,
            ``SoftmaxAttention``).
        d_model (int): Input and output feature dimensionality. Must be divisible by
            ``num_heads``.
        num_heads (int): Number of attention heads.
        d_keys (int, optional): Per-head key/query dimensionality.
            Defaults to ``d_model // num_heads``.
        d_values (int, optional): Per-head value dimensionality.
            Defaults to ``d_model // num_heads``.

    Raises:
        ValueError: If ``d_model`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        num_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        d_keys = d_keys or (d_model // num_heads)
        d_values = d_values or (d_model // num_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * num_heads)
        self.key_projection = nn.Linear(d_model, d_keys * num_heads)
        self.value_projection = nn.Linear(d_model, d_values * num_heads)
        self.out_projection = nn.Linear(d_values * num_heads, d_model)
        self.num_heads = num_heads

    def forward(
        self, queries, keys, values, query_mask=None, key_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply multi-head attention.

        Args:
            queries (torch.Tensor): Shape ``(B, T_1, D)``.
            keys (torch.Tensor): Shape ``(B, T_2, D)``.
            values (torch.Tensor): Shape ``(B, T_2, D)``.
            query_mask (torch.BoolTensor, optional): Shape ``(B, T_1)``. True = valid.
            key_mask (torch.BoolTensor, optional): Shape ``(B, T_2)``. True = valid.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Attended output of shape
                ``(B, T_1, D)`` and optional attention weights.
        """
        B, T_1, _ = queries.shape
        _, T_2, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, T_1, H, -1)
        keys = self.key_projection(keys).view(B, T_2, H, -1)
        values = self.value_projection(values).view(B, T_2, H, -1)

        if query_mask is None and key_mask is None:
            attn_mask = None
        else:
            if query_mask is None:
                query_mask = torch.ones(queries.shape[:2], device=queries.device, dtype=torch.bool)
            if key_mask is None:
                key_mask = torch.ones(keys.shape[:2], device=keys.device, dtype=torch.bool)

            combined_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)  # (B, T_1, T_2)
            attn_mask = torch.full(
                combined_mask.shape, float("-inf"), dtype=queries.dtype, device=queries.device
            )
            attn_mask = attn_mask.masked_fill(combined_mask, 0.0)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T_1, T_2)

        new_values, attn = self.inner_attention(
            queries, keys, values, attn_mask=attn_mask, query_mask=query_mask, key_mask=key_mask
        )
        new_values = new_values.view(B, T_1, -1)

        return self.out_projection(new_values), attn


class BigBirdAttention(nn.Module):
    """BigBird sparse attention: global + local-block + random tokens.

    For self-attention (``tgt_len == src_len``):

    - Global queries (first G positions): full attention over all keys.
    - Non-global queries: each block attends to ``local ∪ global ∪ random`` keys
      with a single softmax — matching the BigBird paper's sparse attention.

    For cross-attention (``tgt_len != src_len``): falls back to full softmax
    attention, as the local-block pattern is undefined across different-length
    sequences.

    Note:
        Random key indices are sampled without duplicates (``torch.randperm``), but
        are not filtered to exclude local-block or global positions. Overlapping
        positions receive slightly higher attention weight. This is a standard
        approximation in BigBird implementations with negligible practical impact.

    Args:
        num_heads (int): Number of attention heads.
        block_size (int): Size of each local attention block.
        num_global_tokens (int): Number of global tokens (first G positions attend everywhere).
        num_random_tokens (int): Number of randomly sampled key positions per block.
        dropout (float): Dropout probability on attention weights. Defaults to ``0.0``.
    """

    def __init__(
        self,
        num_heads: int,
        block_size: int,
        num_global_tokens: int,
        num_random_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, **_kwargs):
        """Compute BigBird sparse attention.

        Args:
            q (torch.Tensor): Queries of shape ``(B, T, H, D)``.
            k (torch.Tensor): Keys of shape ``(B, S, H, D)``.
            v (torch.Tensor): Values of shape ``(B, S, H, D)``.
            attn_mask (torch.Tensor, optional): Additive mask of shape ``(B, 1, T, S)``.
                ``-inf`` at positions to mask out.

        Returns:
            tuple[torch.Tensor, None]: Output of shape ``(B, T, H, D)`` and ``None``
                (no attention weight tensor is returned).
        """
        _, tgt_len, _, head_dim = q.size()
        _, src_len, _, _ = k.size()
        inv_scale = head_dim**-0.5

        G = min(self.num_global_tokens, tgt_len)
        R = self.num_random_tokens
        BS = self.block_size

        attn_output = torch.zeros_like(q)

        if tgt_len == src_len:
            # Random key indices are shared across all non-global blocks.
            # randperm avoids duplicates; slicing gives R distinct positions.
            rand_idx = torch.randperm(src_len, device=q.device)[:R]
            rand_k = k[:, rand_idx]  # (B, R, H, D)
            rand_v = v[:, rand_idx]
            global_k = k[:, :G] if G > 0 else None  # (B, G, H, D)
            global_v = v[:, :G] if G > 0 else None

            # Global queries: full attention over all keys
            if G > 0:
                gq = q[:, :G]  # (B, G, H, D)
                scores = torch.einsum("bghd,bkhd->bhgk", gq, k) * inv_scale  # (B, H, G, src_len)
                if attn_mask is not None:
                    scores += attn_mask[:, :, :G, :]
                w = self.dropout(torch.softmax(scores, dim=-1))
                attn_output[:, :G] = torch.einsum("bhgk,bkhd->bghd", w, v)

            # Non-global queries: one softmax over (local ∪ global ∪ random) keys
            for i in range(G, tgt_len, BS):
                i_end = min(i + BS, tgt_len)
                qb = q[:, i:i_end]  # (B, BL, H, D)

                parts_k = [k[:, i:i_end]]  # local block keys
                parts_v = [v[:, i:i_end]]
                if G > 0:
                    parts_k.append(global_k)
                    parts_v.append(global_v)
                parts_k.append(rand_k)
                parts_v.append(rand_v)

                ck = torch.cat(parts_k, dim=1)  # (B, BL+G+R, H, D)
                cv = torch.cat(parts_v, dim=1)

                scores = torch.einsum("blhd,bkhd->bhlk", qb, ck) * inv_scale  # (B, H, BL, BL+G+R)
                if attn_mask is not None:
                    parts_am = [attn_mask[:, :, i:i_end, i:i_end]]
                    if G > 0:
                        parts_am.append(attn_mask[:, :, i:i_end, :G])
                    parts_am.append(attn_mask[:, :, i:i_end, rand_idx])
                    scores += torch.cat(parts_am, dim=-1)

                w = self.dropout(torch.softmax(scores, dim=-1))
                attn_output[:, i:i_end] = torch.einsum("bhlk,bkhd->blhd", w, cv)

        else:
            # Cross-attention: full softmax — BigBird local-block pattern is undefined
            # across sequences of different lengths.
            scores = torch.einsum("bthd,bshd->bhts", q, k) * inv_scale  # (B, H, tgt_len, src_len)
            if attn_mask is not None:
                scores += attn_mask
            w = self.dropout(torch.softmax(scores, dim=-1))
            attn_output = torch.einsum("bhts,bshd->bthd", w, v)

        return attn_output.contiguous(), None


class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product softmax attention with O(N² D) complexity.

    Computes:

        V' = dropout(softmax(Q Kᵀ / √d)) V

    Args:
        d_model (int): Total model dimensionality.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability on attention weights. Defaults to ``0.0``.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.scale = (d_model // num_heads) ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax attention.

        Args:
            queries (torch.Tensor): Shape ``(B, T_1, H, D)``.
            keys (torch.Tensor): Shape ``(B, T_2, H, D)``.
            values (torch.Tensor): Shape ``(B, T_2, H, D)``.
            attn_mask (torch.Tensor, optional): Additive mask of shape ``(B, 1, T_1, T_2)``.
                ``-inf`` at positions to mask out.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output of shape ``(B, T_1, H, D)``
                and attention weights of shape ``(B, H, T_1, T_2)``.
        """
        scores = torch.einsum("nlhd,nshd->nhls", queries, keys) * self.scale

        if attn_mask is not None:
            scores += attn_mask  # (B, 1, T_1, T_2)

        weights = self.dropout(F.softmax(scores, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", weights, values)
        return V.contiguous(), weights


class LinearAttention(nn.Module):
    """Linear-complexity attention via kernel feature maps — O(N D²).

    Instead of computing the full N×N softmax attention matrix, uses a feature
    map Φ(·) to decompose the kernel and compute:

        V' = normalize(Φ(Q) · Φ(K)ᵀ) · V

    This allows reordering the computation to avoid materializing the attention
    matrix, giving O(N D²) cost where D is the feature-map output dimensionality.

    Masking is handled by zeroing Q at masked query positions and K at masked
    key positions — no NaN risk (unlike softmax with all-``-inf`` rows).

    Attribution:
        Angelos Katharopoulos, Apoorv Vyas — Idiap Research Institute.
        "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
        ICML 2020. https://github.com/idiap/fast-transformers

    Args:
        d_model (int): Total model dimensionality.
        num_heads (int): Number of attention heads (``head_dim = d_model // num_heads``).
        feature_map (callable, optional): Factory that takes ``query_dims`` and returns
            a ``FeatureMap`` instance. Defaults to ``EluFeatureMap`` (elu(x)+1).
    """

    def __init__(self, d_model: int, num_heads: int, feature_map: Callable | None = None):
        super().__init__()
        query_dimensions = d_model // num_heads
        self.feature_map = (
            feature_map(query_dimensions)
            if feature_map is not None
            else EluFeatureMap(query_dimensions)
        )
        self._eps = 1e-6

    def forward(self, queries, keys, values, query_mask=None, key_mask=None, **_kwargs):
        """Compute linear attention.

        Args:
            queries (torch.Tensor): Shape ``(B, T_1, H, D)``.
            keys (torch.Tensor): Shape ``(B, T_2, H, D)``.
            values (torch.Tensor): Shape ``(B, T_2, H, D)``.
            query_mask (torch.BoolTensor, optional): Shape ``(B, T_1)``. True = valid.
            key_mask (torch.BoolTensor, optional): Shape ``(B, T_2)``. True = valid.

        Returns:
            tuple[torch.Tensor, None]: Output of shape ``(B, T_1, H, D)`` and ``None``
                (no attention weight tensor is returned for linear attention).
        """
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)  # (B, T_1, H, D)
        K = self.feature_map.forward_keys(keys)  # (B, T_2, H, D)

        if query_mask is not None:
            Q = Q * query_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T_1) -> (B, T_1, 1, 1)

        if key_mask is not None:
            K = K * key_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T_2) -> (B, T_2, 1, 1)

        KV = torch.einsum("nshd,nshm->nhmd", K, values)  # (B, H, D, d_model)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self._eps)  # (B, T_1, H)
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)  # (B, T_1, H, d_model)

        return V.contiguous(), None


class FeatureMap(nn.Module):
    """Abstract base class defining the feature map interface for linear attention.

    Subclasses implement Φ(·) such that Φ(Q)ᵀΦ(K) approximates (or equals)
    the desired attention kernel.

    Args:
        query_dims (int): Head dimensionality (``d_model // n_heads``).
    """

    def __init__(self, query_dims: int):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device: torch.device) -> None:
        """Reinitialize (re-sample) the feature map parameters for this forward pass.

        Called once per forward pass by ``LinearAttention``. For random feature maps
        this samples a fresh projection matrix.

        Args:
            device (torch.device): The torch device to create tensors on.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    def forward_queries(self, x: torch.Tensor) -> torch.Tensor:
        """Encode queries using this feature map.

        Args:
            x (torch.Tensor): Query tensor of shape ``(B, T, H, D)``.

        Returns:
            torch.Tensor: Encoded queries of the same leading shape.
        """
        return self(x)

    def forward_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Encode keys using this feature map.

        Args:
            x (torch.Tensor): Key tensor of shape ``(B, T, H, D)``.

        Returns:
            torch.Tensor: Encoded keys of the same leading shape.
        """
        return self(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` using this feature map.

        For symmetric feature maps it suffices to define this method. For
        asymmetric maps, override ``forward_queries`` and ``forward_keys``
        separately.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded output.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs) -> Callable:
        """Return a factory callable for constructing this feature map.

        The returned callable accepts ``query_dims`` and returns an instance of
        this class. Inherited by all subclasses, enabling use with
        ``LinearAttention``'s ``feature_map`` argument.

        Returns:
            Callable[[int], FeatureMap]: A factory function ``query_dims → instance``.
        """

        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)

        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Feature map defined by an element-wise activation function.

    Args:
        query_dims (int): Head dimensionality.
        activation_function (callable): Applied element-wise to the input tensor.
    """

    def __init__(self, query_dims: int, activation_function: Callable):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device: torch.device) -> None:  # noqa: ARG002
        """No-op: activation-based feature maps have no random parameters."""
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the activation function element-wise.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Activated tensor of the same shape.
        """
        return self.activation_function(x)


class EluFeatureMap(ActivationFunctionFeatureMap):
    """ELU+1 feature map — default for ``LinearAttention``.

    Implements Φ(x) = elu(x) + 1, which satisfies Φ(x) ≥ 0 everywhere and
    yields a valid positive-definite kernel without random projections.

    Args:
        query_dims (int): Head dimensionality (``d_model // n_heads``).
    """

    def __init__(self, query_dims: int):
        super().__init__(query_dims, lambda x: F.elu(x) + 1)


class PerformerFeatureMap(FeatureMap):
    """performer (FAVOR+) positive random feature map (Choromanski et al., ICLR 2021).

    Provides an unbiased estimator of the softmax attention kernel using
    orthogonal random features. Unlike ``elu+1``, uses ``r >> head_dim`` features,
    directly addressing the capacity limitation of small head dimensions:

        Φ(x)ᵢ = exp(x·ωᵢ − ‖x‖²/2) / √r   for r orthogonal vectors ωᵢ

    E[Φ(x)ᵀΦ(y)] ≈ exp(xᵀy) — unbiased estimator of the softmax kernel.
    ``new_feature_map()`` resamples the projection each forward pass, reducing
    variance across training steps.

    Select via config: ``attention_type: performer``

    Tune via config:   ``performer_num_random_features: 64``  (default: ``max(head_dim*4, 32)``)

    Args:
        query_dims (int): Head dimensionality (``d_model // n_heads``).
        num_features (int, optional): Number of random features ``r``.
            Defaults to ``max(query_dims * 4, 32)``.
    """

    def __init__(self, query_dims: int, num_features: int | None = None):
        super().__init__(query_dims)
        self.num_features = num_features if num_features is not None else max(query_dims * 4, 32)
        self.projection: torch.Tensor | None = None

    def new_feature_map(self, device: torch.device) -> None:
        """Sample a new orthogonal random projection matrix.

        Args:
            device (torch.device): Target device for the projection tensor.
        """
        d, r = self.query_dims, self.num_features
        num_blocks = math.ceil(r / d)
        blocks = [self._orthogonal_block(d, device) for _ in range(num_blocks)]
        self.projection = torch.cat(blocks, dim=1)[:, :r]  # (d, r)

    def _orthogonal_block(self, d: int, device: torch.device) -> torch.Tensor:
        """Draw a random orthonormal matrix scaled to match Gaussian vector norms."""
        G = torch.randn(d, d, device=device)
        Q, _ = torch.linalg.qr(G)  # orthonormal columns
        return Q * (d**0.5)  # scale to match expected Gaussian vector norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the FAVOR+ feature map.

        Args:
            x (torch.Tensor): Input of shape ``(..., query_dims)``.

        Returns:
            torch.Tensor: Positive random features of shape ``(..., num_features)``.
        """
        if self.projection is None:
            raise RuntimeError("call new_feature_map() before forward()")
        x_proj = x @ self.projection  # (..., num_features)
        norm_sq = x.square().sum(-1, keepdim=True) * 0.5  # (..., 1)
        return torch.exp(x_proj - norm_sq) / (self.num_features**0.5)


class GatedAttentionUnit(nn.Module):
    """flash (GAU) — Hua et al., ICML 2022.

    Replaces multi-head attention with single-head gated linear attention:

        u = SiLU(W_u · queries)          # gate from query stream
        v = W_v · values                 # values from key/value stream
        q = relu(W_q · queries)²         # scoring query (always ≥ 0)
        k = relu(W_k · keys)²            # scoring key (always ≥ 0)
        a = linear_attn(q, k, v)         # O(N·s) single-head attention
        output = W_o · (u ⊙ a)          # gated output

    relu² ensures k·q ≥ 0 everywhere, keeping the linear attention denominator
    positive without a learned feature map. Supports cross-attention: gate and
    scoring query come from the query (target) stream; scoring key and value
    come from the key/value (source) stream.

    The forward interface matches ``AttentionLayer``, so it is a drop-in
    replacement in ``TransformerEncoderLayer`` without any changes to
    ``transformer.py``.

    Select via config: ``attention_type: flash``

    Tune via config:   ``flash_query_key_dim: 32``  (default: ``max(d_model // 2, 16)``)
                       ``dropout_attention: 0.1``

    Args:
        d_model (int): Input and output feature dimensionality.
        query_key_dim (int, optional): Scoring dimension ``s``.
            Defaults to ``max(d_model // 2, 16)``.
        dropout (float): Dropout on the gated pre-projection tensor ``u ⊙ a``.
            Defaults to ``0.0``.
    """

    def __init__(
        self,
        d_model: int,
        query_key_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.query_key_dim = query_key_dim if query_key_dim is not None else max(d_model // 2, 16)
        self._eps = 1e-6

        self.gate_proj = nn.Linear(d_model, d_model)  # u gate from query
        self.value_proj = nn.Linear(d_model, d_model)  # v from key/value stream
        self.query_score_proj = nn.Linear(d_model, self.query_key_dim)  # scoring q
        self.key_score_proj = nn.Linear(d_model, self.query_key_dim)  # scoring k
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Compute gated linear attention.

        Args:
            queries (torch.Tensor): Shape ``(B, T_q, d_model)``.
            keys (torch.Tensor): Shape ``(B, T_k, d_model)``.
            values (torch.Tensor): Shape ``(B, T_k, d_model)``.
            query_mask (torch.BoolTensor, optional): Shape ``(B, T_q)``. True = valid.
                Masked output positions are set to zero.
            key_mask (torch.BoolTensor, optional): Shape ``(B, T_k)``. True = valid.
                Masked key/value positions are zeroed before accumulation.

        Returns:
            tuple[torch.Tensor, None]: Output of shape ``(B, T_q, d_model)`` and
                ``None`` (no attention weight tensor is returned).
        """
        u = F.silu(self.gate_proj(queries))  # (B, T_q, d_model)
        v = self.value_proj(values)  # (B, T_k, d_model)
        q = F.relu(self.query_score_proj(queries)) ** 2  # (B, T_q, s)
        k = F.relu(self.key_score_proj(keys)) ** 2  # (B, T_k, s)

        if key_mask is not None:
            km = key_mask.unsqueeze(-1)  # (B, T_k, 1)
            k = k * km
            v = v * km

        # Single-head linear attention: O(N·s)
        kv = torch.einsum("bks,bkd->bsd", k, v)  # (B, s, d_model)
        k_sum = k.sum(dim=1)  # (B, s)
        qkv = torch.einsum("bqs,bsd->bqd", q, kv)  # (B, T_q, d_model)
        z = torch.einsum("bqs,bs->bq", q, k_sum) + self._eps  # (B, T_q)
        a = qkv / z.unsqueeze(-1)  # (B, T_q, d_model)

        out = self.out_proj(self.dropout(u * a))  # (B, T_q, d_model)

        if query_mask is not None:
            out = out * query_mask.unsqueeze(-1)

        return out, None
