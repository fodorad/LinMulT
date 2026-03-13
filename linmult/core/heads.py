"""Output head types, factory, and HeadModule for LinMulT/LinT models."""

import torch
import torch.nn as nn

from linmult.core.config import HeadConfig
from linmult.core.norm import BN, IN
from linmult.core.temporal import AttentionPooling, GlobalAvgPooling, GlobalMaxPooling


def _create_norm(norm_type: str, feature_dim: int, *, time_aware: bool) -> nn.Module:
    """Create a normalization layer from a type string.

    Args:
        norm_type: One of ``"bn"`` or ``"in"``.
        feature_dim: Number of features to normalize.
        time_aware: Whether the input has a time dimension.

    Returns:
        A ``BN`` or ``IN`` module.

    Raises:
        ValueError: If ``norm_type`` is not recognized.
    """
    if norm_type == "bn":
        return BN(feature_dim, time_aware=time_aware)
    if norm_type == "in":
        return IN(feature_dim, time_aware=time_aware)
    raise ValueError(f"Unknown norm type: {norm_type!r}. Choose from {{'bn', 'in'}}.")


def _create_pooling(pooling_type: str, feature_dim: int) -> nn.Module:
    """Create a pooling layer from a type string.

    Args:
        pooling_type: One of ``"gap"``, ``"gmp"``, or ``"attentionpool"``.
        feature_dim: Number of features (used for attention pool hidden dim).

    Returns:
        A pooling module.

    Raises:
        ValueError: If ``pooling_type`` is not recognized.
    """
    if pooling_type == "gap":
        return GlobalAvgPooling()
    if pooling_type == "gmp":
        return GlobalMaxPooling()
    if pooling_type == "attentionpool":
        return AttentionPooling(feature_dim)
    raise ValueError(
        f"Unknown pooling type: {pooling_type!r}. Choose from {{'gap', 'gmp', 'attentionpool'}}."
    )


class BaseHead(nn.Module):
    """Abstract base class for all output heads.

    Subclasses must implement :meth:`forward`. Use :meth:`from_config` as the
    standard factory entry point; it simply delegates to ``__init__``.

    Args:
        _input_dim (int): Input feature dimensionality (stored for subclass use).
        _output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration.
    """

    def __init__(self, _input_dim: int, _output_dim: int, config: HeadConfig):
        """Initialize BaseHead."""
        super().__init__()
        self.name = config.name if config.name else self.__class__.__name__
        self.config = config

    def extra_repr(self) -> str:
        """Return the head name for identification in repr output."""
        return f"name={self.name!r}"  # pragma: no cover

    @classmethod
    def from_config(cls, input_dim: int, output_dim: int, config: HeadConfig) -> "BaseHead":
        """Construct a head from keyword arguments.

        Args:
            input_dim (int): Input feature dimensionality.
            output_dim (int): Output feature dimensionality.
            config (HeadConfig): Head configuration.

        Returns:
            BaseHead: A new instance of this head class.
        """
        return cls(input_dim, output_dim, config)


class SequenceAggregationHead(BaseHead):
    """Output head that aggregates a sequence to a single vector.

    Maps ``(B, T, F)`` → ``(B, output_dim)`` by normalizing, projecting to a
    hidden dimension, pooling along the time axis, and projecting to the output
    dimension.

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Relevant attributes:

            - ``norm`` (str): Normalisation type, ``"bn"`` or ``"in"``. Default ``"bn"``.
            - ``pooling`` (str): Pooling type, ``"gap"``, ``"gmp"``, or
              ``"attentionpool"``. Default ``"gap"``.
            - ``hidden_dim`` (int): Hidden projection size. Default ``256``.
            - ``dropout`` (float): Dropout in the first projection. Default ``0.1``.
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize SequenceAggregationHead."""
        super().__init__(input_dim, output_dim, config)

        self.norm = _create_norm(config.norm, input_dim, time_aware=True)

        pooling = config.pooling if config.pooling is not None else "gap"
        self.pool: nn.Module = _create_pooling(pooling, config.hidden_dim)

        self.proj_1 = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.proj_2 = nn.Linear(config.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Aggregate and project.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, F)``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T)``. True = valid.

        Returns:
            torch.Tensor: Output of shape ``(B, output_dim)``.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x)  # (B, T, F)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj_1(x)
        x = self.pool(x, mask)  # (B, T, F) -> (B, F)
        x = self.proj_2(x)

        return x


class SequenceHead(BaseHead):
    """Output head that preserves the time dimension.

    Maps ``(B, T, F)`` → ``(B, T, output_dim)`` by normalizing and projecting
    each timestep independently.

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Relevant attributes:

            - ``norm`` (str): Normalisation type, ``"bn"`` or ``"in"``. Default ``"bn"``.
            - ``hidden_dim`` (int): Hidden projection size. Default ``256``.
            - ``dropout`` (float): Dropout in the projection. Default ``0.1``.
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize SequenceHead."""
        super().__init__(input_dim, output_dim, config)

        self.norm = _create_norm(config.norm, input_dim, time_aware=True)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize and project each timestep.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, F)``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T)``. True = valid.

        Returns:
            torch.Tensor: Output of shape ``(B, T, output_dim)``.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x)
        x = self.proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


class VectorHead(BaseHead):
    """Output head for vector (already-aggregated) inputs.

    Maps ``(B, F)`` → ``(B, output_dim)`` by normalizing and projecting.

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Relevant attributes:

            - ``norm`` (str): Normalisation type, ``"bn"`` or ``"in"``. Default ``"bn"``.
            - ``hidden_dim`` (int): Hidden projection size. Default ``256``.
            - ``dropout`` (float): Dropout in the projection. Default ``0.1``.
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize VectorHead."""
        super().__init__(input_dim, output_dim, config)

        self.norm = _create_norm(config.norm, input_dim, time_aware=False)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        """Normalize and project a vector.

        Args:
            x (torch.Tensor): Input of shape ``(B, F)``.

        Returns:
            torch.Tensor: Output of shape ``(B, output_dim)``.
        """
        x = self.norm(x)
        x = self.proj(x)
        return x


class SimpleHead(BaseHead):
    """Lightweight linear head with optional time-dimension pooling.

    Applies an optional pooling step followed by a single linear projection.
    Depending on the ``pooling`` config attribute, the mapping is:

    - No pooling (``None``): ``(B, T, F)`` → ``(B, T, output_dim)``
    - With pooling (``"gap"`` / ``"gmp"`` / ``"attentionpool"``):
        ``(B, T, F)`` → ``(B, output_dim)``

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Relevant attribute:

            - ``pooling`` (str, optional): One of ``"gap"``, ``"gmp"``,
              ``"attentionpool"``, or ``None`` (no pooling).
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize SimpleHead."""
        super().__init__(input_dim, output_dim, config)

        self.pool: nn.Module | None = (
            _create_pooling(config.pooling, input_dim) if config.pooling else None
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_kwargs) -> torch.Tensor:
        """Apply optional pooling then linear projection.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, F)`` or ``(B, F)``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T)``. True = valid.
                Passed through to pooling layers when ``pool`` is configured.

        Returns:
            torch.Tensor: Output of shape ``(B, output_dim)`` if pooled,
                otherwise ``(B, T, output_dim)``.
        """
        if self.pool is not None:
            x = self.pool(x, mask)  # (B, T, F) -> (B, F)
        return self.linear(x)


class UpsampleHead(BaseHead):
    """Output head with learnable temporal upsampling.

    Maps ``(B, T_in, F)`` → ``(B, output_time_dim, output_dim)`` by projecting
    the feature dimension, applying a stack of transposed convolutions (each
    doubling the time axis), then a final adaptive pool to hit the exact target.

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Required attributes:

            - ``output_time_dim`` (int): Target time dimension.
            - ``input_time_dim`` (int): Source time dimension.
            - ``dropout`` (float): Dropout probability. Default ``0.1``.
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize UpsampleHead."""
        super().__init__(input_dim, output_dim, config)
        if config.input_time_dim is None:
            raise ValueError("UpsampleHead requires 'input_time_dim' to be set in HeadConfig.")
        if config.output_time_dim is None:
            raise ValueError("UpsampleHead requires 'output_time_dim' to be set in HeadConfig.")
        self.target_time_dim = config.output_time_dim
        input_time_dim = config.input_time_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.upsample_layers = nn.ModuleList()
        current_dim = input_time_dim

        while current_dim * 2 <= self.target_time_dim:
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=output_dim,
                        out_channels=output_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.GELU(),
                )
            )
            current_dim *= 2

        self.final_adjust = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
            nn.AdaptiveAvgPool1d(self.target_time_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Upsample and project.

        Args:
            x (torch.Tensor): Input of shape ``(B, T_in, F)``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T_in)``. True = valid.
                Masked positions are zeroed before processing.

        Returns:
            torch.Tensor: Output of shape ``(B, output_time_dim, output_dim)``.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj(x)  # (B, T_in, output_dim)
        x = x.transpose(1, 2)  # (B, output_dim, T_in)

        for layer in self.upsample_layers:
            x = layer(x)

        x = self.final_adjust(x)
        return x.transpose(1, 2)  # (B, output_time_dim, output_dim)


class DownsampleHead(BaseHead):
    """Output head with learnable temporal downsampling.

    Maps ``(B, T_in, F)`` → ``(B, output_time_dim, output_dim)`` by projecting
    the feature dimension, applying strided convolutions (each halving the time
    axis), then a final adaptive average pool to hit the exact target.

    Args:
        input_dim (int): Input feature dimensionality.
        output_dim (int): Output feature dimensionality.
        config (HeadConfig): Head configuration. Required attributes:

            - ``output_time_dim`` (int): Target time dimension.
            - ``input_time_dim`` (int): Source time dimension.
            - ``dropout`` (float): Dropout probability. Default ``0.1``.
    """

    def __init__(self, input_dim: int, output_dim: int, config: HeadConfig):
        """Initialize DownsampleHead."""
        super().__init__(input_dim, output_dim, config)
        if config.input_time_dim is None:
            raise ValueError("DownsampleHead requires 'input_time_dim' to be set in HeadConfig.")
        if config.output_time_dim is None:
            raise ValueError("DownsampleHead requires 'output_time_dim' to be set in HeadConfig.")
        self.target_time_dim = config.output_time_dim
        input_time_dim = config.input_time_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        self.downsample_layers = nn.ModuleList()
        current_dim = input_time_dim

        while current_dim // 2 >= self.target_time_dim:
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=output_dim,
                        out_channels=output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode="replicate",
                    ),
                    nn.GELU(),
                )
            )
            current_dim = current_dim // 2

        self.final_pool = nn.AdaptiveAvgPool1d(self.target_time_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Downsample and project.

        Args:
            x (torch.Tensor): Input of shape ``(B, T_in, F)``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T_in)``. True = valid.
                Masked positions are zeroed before processing.

        Returns:
            torch.Tensor: Output of shape ``(B, output_time_dim, output_dim)``.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj(x)  # (B, T_in, output_dim)
        x = x.transpose(1, 2)  # (B, output_dim, T_in)

        for layer in self.downsample_layers:
            x = layer(x)

        x = self.final_pool(x)

        return x.transpose(1, 2)  # (B, output_time_dim, output_dim)


class HeadFactory:
    """Registry and factory for output head types.

    New head classes can be registered at runtime with :meth:`register_head`,
    then instantiated by name with :meth:`create_head`.

    Built-in types: ``"sequence_aggregation"``, ``"sequence"``, ``"vector"``,
    ``"simple"``, ``"upsample"``, ``"downsample"``.
    """

    _registry: dict[str, type[BaseHead]] = {
        "sequence_aggregation": SequenceAggregationHead,
        "sequence": SequenceHead,
        "vector": VectorHead,
        "simple": SimpleHead,
        "upsample": UpsampleHead,
        "downsample": DownsampleHead,
    }

    @classmethod
    def register_head(cls, name: str, head_cls: type[BaseHead]) -> None:
        """Register a custom head class under a given name.

        Args:
            name (str): Registry key used in ``config["type"]``.
            head_cls (type[BaseHead]): Head class to register.
        """
        cls._registry[name] = head_cls

    @classmethod
    def create_head(
        cls, type: str, input_dim: int, output_dim: int, config: HeadConfig
    ) -> BaseHead:
        """Instantiate a registered head by type name.

        Args:
            type (str): Registered head type name.
            input_dim (int): Input feature dimensionality.
            output_dim (int): Output feature dimensionality.
            config (HeadConfig): Head configuration.

        Returns:
            BaseHead: The constructed head module.

        Raises:
            ValueError: If ``type`` is not registered.
        """
        if type not in cls._registry:
            raise ValueError(f"Unknown head type: {type}. Registered: {list(cls._registry.keys())}")

        return cls._registry[type].from_config(
            input_dim=input_dim, output_dim=output_dim, config=config
        )


class HeadModule(nn.Module):
    """Self-contained output head container.

    Builds all output heads from a list of :class:`HeadConfig` using
    :class:`HeadFactory`, and applies them in the forward pass.

    Args:
        input_dim: Input feature dimension fed to each head.
        head_configs: List of head configurations.
    """

    def __init__(self, input_dim: int, head_configs: list[HeadConfig]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict()
        for i, cfg in enumerate(head_configs):
            head = HeadFactory.create_head(
                type=cfg.type,
                input_dim=input_dim,
                output_dim=cfg.output_dim,
                config=cfg,
            )
            name = cfg.name if cfg.name else f"head_{i}"
            self.heads[name] = head

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Apply all heads to the input.

        Args:
            x: Input tensor ``(B, [T,] input_dim)``.
            mask: Optional boolean mask ``(B, T)``.

        Returns:
            Dict mapping head name to output tensor.
        """
        return {name: head(x, mask=mask) for name, head in self.heads.items()}
