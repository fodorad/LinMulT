"""Typed configuration dataclasses for LinT and LinMulT."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from linmult.core.attention import AttentionConfig


@dataclass
class HeadConfig:
    """Configuration for one output head.

    Args:
        type (str): Head type. One of ``"sequence_aggregation"``, ``"sequence"``,
            ``"vector"``, ``"simple"``, ``"upsample"``, ``"downsample"``.
        output_dim (int): Output feature dimensionality.
        name (str): Head name used as key in the output dict. Defaults to ``""``
            (resolved to the head class name at construction time).
        norm (str): Normalisation type for heads that use it. One of ``"bn"``, ``"in"``.
            Defaults to ``"bn"``.
        pooling (str | None): Pooling strategy. One of ``"gap"``, ``"gmp"``,
            ``"attentionpool"``, or ``None`` (no pooling, e.g. for :class:`SimpleHead`
            without temporal reduction). Defaults to ``None`` (preserve sequence).
        hidden_dim (int): Hidden projection size. Defaults to ``256``.
        dropout (float): Dropout probability used inside the head. Defaults to ``0.1``.
        input_time_dim (int | None): Source time dimension for
            :class:`UpsampleHead` / :class:`DownsampleHead`. Defaults to ``None``.
        output_time_dim (int | None): Target time dimension for
            :class:`UpsampleHead` / :class:`DownsampleHead`. Defaults to ``None``.
    """

    type: str
    output_dim: int
    name: str = ""
    # SequenceAggregation / Sequence / Vector / SimpleHead
    norm: str = "bn"
    pooling: str | None = None
    hidden_dim: int = 256
    dropout: float = 0.1
    # Upsample / Downsample
    input_time_dim: int | None = None
    output_time_dim: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeadConfig:
        """Construct from a plain dict, ignoring unknown keys.

        Args:
            d (dict): Dictionary of head configuration values.

        Returns:
            HeadConfig: A new :class:`HeadConfig` instance.
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


def _heads_from_list(raw: list[Any]) -> list[HeadConfig]:
    return [h if isinstance(h, HeadConfig) else HeadConfig.from_dict(h) for h in raw]


@dataclass
class LinTConfig:
    """Configuration for :class:`LinT` (unimodal linear-complexity transformer).

    **Required**

    Args:
        input_feature_dim (int): Input feature dimensionality.

    **Identity**

    Args:
        name (str): Model name shown in ``repr``. Defaults to ``""``.

    **Core**

    Args:
        d_model (int): Internal embedding dimension. Defaults to ``40``.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        cmt_num_layers (int): Self-attention encoder depth. Defaults to ``6``.

    **Attention**

    Args:
        attention_type (str): Attention mechanism. One of ``"linear"`` (default),
            ``"performer"``, ``"flash"``, ``"softmax"``, ``"bigbird"``, ``"mha"``.
        flash_query_key_dim (int | None): Scoring dimension for ``"flash"`` (GAU).
            Defaults to ``None`` (computed as ``max(d_model // 2, 16)``).
        performer_num_random_features (int | None): Random feature count for
            ``"performer"``. Defaults to ``None`` (computed as ``max(head_dim * 4, 32)``).
        bigbird_block_size (int): Local block size for ``"bigbird"``. Defaults to ``64``.
        bigbird_num_global_tokens (int): Global tokens for ``"bigbird"``. Defaults to ``16``.
        bigbird_num_random_tokens (int): Random tokens for ``"bigbird"``. Defaults to ``10``.

    **Dropout**

    Args:
        dropout_input (float): Dropout on input before projection. Defaults to ``0.0``.
        dropout_output (float): FFN-fusion output dropout. Defaults to ``0.0``.
        dropout_pe (float): Dropout after positional encoding. Defaults to ``0.0``.
        dropout_ffn (float): Dropout in transformer FFN. Defaults to ``0.1``.
        dropout_attention (float): Attention-weight dropout. Defaults to ``0.0``.

    **TRM**

    Args:
        time_dim_reducer (str | None): Collapse ``(B, T, F)`` → ``(B, F)`` before
            heads. One of ``"attentionpool"``, ``"gap"``, ``"gmp"``, ``"last"``, or
            ``None`` (no reduction). Defaults to ``None``.

    **Optional modules**

    Args:
        add_module_ffn_fusion (bool): FFN + residual block after the encoder.
            Defaults to ``False``.

    **Heads**

    Args:
        heads (list[HeadConfig | dict]): Output head configurations. Plain dicts
            are automatically coerced to :class:`HeadConfig`. Defaults to ``[]``.

    **Special handling**

    Args:
        special_handling (dict[str, Any]): Modality-specific input handling
            (e.g. weighted-sum of transformer layers). Defaults to ``{}``.
    """

    # --- Required ---
    input_feature_dim: int

    # --- Identity ---
    name: str = ""

    # --- Core ---
    d_model: int = 40
    num_heads: int = 8
    cmt_num_layers: int = 6

    # --- Attention ---
    attention_type: str = "linear"
    flash_query_key_dim: int | None = None
    performer_num_random_features: int | None = None
    bigbird_block_size: int = 64
    bigbird_num_global_tokens: int = 16
    bigbird_num_random_tokens: int = 10

    # --- Dropout ---
    dropout_input: float = 0.0
    dropout_output: float = 0.0
    dropout_pe: float = 0.0
    dropout_ffn: float = 0.1
    dropout_attention: float = 0.0

    # --- TRM ---
    time_dim_reducer: str | None = None

    # --- Optional modules ---
    add_module_ffn_fusion: bool = False

    # --- Heads ---
    heads: list[HeadConfig] = field(default_factory=list)

    # --- Special handling ---
    special_handling: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce head dicts to :class:`HeadConfig` instances."""
        self.heads = _heads_from_list(self.heads)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinTConfig:
        """Construct from a plain dict (e.g. loaded from YAML), ignoring unknown keys.

        Args:
            d (dict): Dictionary of configuration values.

        Returns:
            LinTConfig: A new :class:`LinTConfig` instance.
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> LinTConfig:
        """Load a :class:`LinTConfig` from a YAML file.

        Args:
            path (str | Path): Path to the YAML configuration file.

        Returns:
            LinTConfig: A new :class:`LinTConfig` instance.
        """
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def build_attention_config(self) -> AttentionConfig:
        """Build an :class:`~linmult.core.attention.AttentionConfig` from this config.

        Returns:
            AttentionConfig: Attention configuration ready for use in model construction.
        """
        from linmult.core.attention import AttentionConfig

        return AttentionConfig(
            type=self.attention_type,
            dropout=self.dropout_attention,
            flash_query_key_dim=self.flash_query_key_dim,
            performer_num_random_features=self.performer_num_random_features,
            bigbird_block_size=self.bigbird_block_size,
            bigbird_num_global_tokens=self.bigbird_num_global_tokens,
            bigbird_num_random_tokens=self.bigbird_num_random_tokens,
        )


@dataclass
class LinMulTConfig:
    """Configuration for :class:`LinMulT` (multimodal linear-complexity transformer).

    **Required**

    Args:
        input_feature_dim (list[int]): Input feature dimensionality per modality.
            Must have at least 2 entries.

    **Identity**

    Args:
        name (str): Model name shown in ``repr``. Defaults to ``""``.

    **Core**

    Args:
        d_model (int): Internal embedding dimension. Defaults to ``40``.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        cmt_num_layers (int): Cross-modal transformer (CMT) encoder depth. Defaults to ``6``.
        branch_sat_num_layers (int): Per-branch self-attention encoder depth. Defaults to ``6``.

    **Attention**

    Args:
        attention_type (str): Attention mechanism. One of ``"linear"`` (default),
            ``"performer"``, ``"flash"``, ``"softmax"``, ``"bigbird"``, ``"mha"``.
        flash_query_key_dim (int | None): Scoring dimension for ``"flash"`` (GAU).
            Defaults to ``None`` (computed as ``max(d_model // 2, 16)``).
        performer_num_random_features (int | None): Random feature count for
            ``"performer"``. Defaults to ``None`` (computed as ``max(head_dim * 4, 32)``).
        bigbird_block_size (int): Local block size for ``"bigbird"``. Defaults to ``64``.
        bigbird_num_global_tokens (int): Global tokens for ``"bigbird"``. Defaults to ``16``.
        bigbird_num_random_tokens (int): Random tokens for ``"bigbird"``. Defaults to ``10``.

    **Dropout**

    Args:
        dropout_input (float): Dropout on input before projection. Defaults to ``0.0``.
        dropout_output (float): FFN-fusion output dropout. Defaults to ``0.0``.
        dropout_pe (float): Dropout after positional encoding. Defaults to ``0.0``.
        dropout_ffn (float): Dropout in transformer FFN. Defaults to ``0.1``.
        dropout_attention (float): Attention-weight dropout. Defaults to ``0.0``.
        dropout_tam (float): Dropout inside the TAM projector. Defaults to ``0.1``.

    **Unimodal self-attention (optional)**

    Args:
        add_module_unimodal_sat (bool): Per-modality self-attention transformer (SAT)
            before cross-modal layers. Defaults to ``False``.
        unimodal_sat_num_layers (int): Unimodal SAT encoder depth. Defaults to ``6``.

    **Multimodal signal via TAM (optional)**

    Args:
        add_module_multimodal_signal (bool): Prepend a TAM-fused cross-modal summary
            to each branch. Requires ``tam_time_dim``. Defaults to ``False``.
        mms_num_layers (int): Encoder depth inside the MMS TAM. Defaults to ``6``.
        tam_aligner (str | None): Temporal alignment strategy. One of ``"aap"``,
            ``"amp"``, ``"padding"``. Required when either TAM module is enabled.
            Defaults to ``None``.
        tam_time_dim (int | None): Target time dimension after TAM alignment.
            Required when either TAM module is enabled. Defaults to ``None``.

    **TRM**

    Args:
        time_dim_reducer (str | None): Collapse ``(B, T, F)`` → ``(B, F)`` before
            heads. One of ``"attentionpool"``, ``"gap"``, ``"gmp"``, ``"last"``, or
            ``None`` (no reduction). Defaults to ``None``.

    **Fusion (optional)**

    Args:
        add_module_tam_fusion (bool): TAM-based fusion after cross-modal branches.
            Requires ``tam_time_dim``. Defaults to ``False``.
        fusion_num_layers (int): Encoder depth inside the TAM fusion module. Defaults to ``6``.
        add_module_sat_fusion (bool): Self-attention transformer on the fused representation.
            Defaults to ``False``.
        fusion_sat_num_layers (int): Fusion SAT encoder depth. Defaults to ``6``.
        add_module_ffn_fusion (bool): FFN + residual block after fusion. Defaults to ``False``.

    **Heads**

    Args:
        heads (list[HeadConfig | dict]): Output head configurations. Plain dicts
            are automatically coerced to :class:`HeadConfig`. Defaults to ``[]``.
        auxiliary_heads (list[HeadConfig | dict]): Per-branch auxiliary head configs.
            Plain dicts are automatically coerced to :class:`HeadConfig`. Defaults to ``[]``.

    **Special handling**

    Args:
        special_handling (dict[str, Any]): Modality-specific input handling
            (e.g. weighted-sum of transformer layers). Defaults to ``{}``.
    """

    # --- Required ---
    input_feature_dim: list[int]

    # --- Identity ---
    name: str = ""

    # --- Core ---
    d_model: int = 40
    num_heads: int = 8
    cmt_num_layers: int = 6
    branch_sat_num_layers: int = 6

    # --- Attention ---
    attention_type: str = "linear"
    flash_query_key_dim: int | None = None
    performer_num_random_features: int | None = None
    bigbird_block_size: int = 64
    bigbird_num_global_tokens: int = 16
    bigbird_num_random_tokens: int = 10

    # --- Dropout ---
    dropout_input: float = 0.0
    dropout_output: float = 0.0
    dropout_pe: float = 0.0
    dropout_ffn: float = 0.1
    dropout_attention: float = 0.0
    dropout_tam: float = 0.1

    # --- Unimodal self-attention (optional) ---
    add_module_unimodal_sat: bool = False
    unimodal_sat_num_layers: int = 6

    # --- Multimodal signal via TAM (optional) ---
    add_module_multimodal_signal: bool = False
    mms_num_layers: int = 6
    tam_aligner: str | None = None
    tam_time_dim: int | None = None

    # --- TRM ---
    time_dim_reducer: str | None = None

    # --- Fusion (optional) ---
    add_module_tam_fusion: bool = False
    fusion_num_layers: int = 6
    add_module_sat_fusion: bool = False
    fusion_sat_num_layers: int = 6
    add_module_ffn_fusion: bool = False

    # --- Heads ---
    heads: list[HeadConfig] = field(default_factory=list)
    auxiliary_heads: list[HeadConfig] = field(default_factory=list)

    # --- Special handling ---
    special_handling: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce head dicts and validate TAM-dependent options."""
        self.heads = _heads_from_list(self.heads)
        self.auxiliary_heads = _heads_from_list(self.auxiliary_heads)

        needs_tam = self.add_module_multimodal_signal or self.add_module_tam_fusion
        if needs_tam and not self.tam_time_dim:
            modules = [
                name
                for flag, name in [
                    (self.add_module_multimodal_signal, "add_module_multimodal_signal"),
                    (self.add_module_tam_fusion, "add_module_tam_fusion"),
                ]
                if flag
            ]
            raise ValueError(
                f"{', '.join(modules)} require 'tam_time_dim' to be set to a positive int."
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LinMulTConfig:
        """Construct from a plain dict (e.g. loaded from YAML), ignoring unknown keys.

        Args:
            d (dict): Dictionary of configuration values.

        Returns:
            LinMulTConfig: A new :class:`LinMulTConfig` instance.
        """
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> LinMulTConfig:
        """Load a :class:`LinMulTConfig` from a YAML file.

        Args:
            path (str | Path): Path to the YAML configuration file.

        Returns:
            LinMulTConfig: A new :class:`LinMulTConfig` instance.
        """
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def build_attention_config(self) -> AttentionConfig:
        """Build an :class:`~linmult.core.attention.AttentionConfig` from this config.

        Returns:
            AttentionConfig: Attention configuration ready for use in model construction.
        """
        from linmult.core.attention import AttentionConfig

        return AttentionConfig(
            type=self.attention_type,
            dropout=self.dropout_attention,
            flash_query_key_dim=self.flash_query_key_dim,
            performer_num_random_features=self.performer_num_random_features,
            bigbird_block_size=self.bigbird_block_size,
            bigbird_num_global_tokens=self.bigbird_num_global_tokens,
            bigbird_num_random_tokens=self.bigbird_num_random_tokens,
        )
