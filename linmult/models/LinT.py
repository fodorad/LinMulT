"""LinT: unimodal linear-complexity transformer."""

from pathlib import Path

import torch
from torch import nn

from linmult.core.config import LinTConfig
from linmult.core.ffn import FFNResidual
from linmult.core.heads import HeadModule
from linmult.core.projection import ProjectionModule
from linmult.core.temporal import TRM
from linmult.core.transformer import TransformerEncoder


class LinT(nn.Module):
    """Linear-complexity Transformer for a single input modality.

    Processes one time-series input through a projection + self-attention
    pipeline and applies configurable output heads.

    Args:
        config (LinTConfig | str | Path): Configuration object or path to a YAML file.
    """

    def __init__(self, config: LinTConfig | str | Path):
        """Initialize LinT.

        Args:
            config (LinTConfig | str | Path): Configuration object or path to a YAML
                file. When a path is given the file is loaded with
                :meth:`LinTConfig.from_yaml`.
        """
        super().__init__()

        if isinstance(config, str | Path):
            config = LinTConfig.from_yaml(config)

        if not isinstance(config.input_feature_dim, int):
            raise ValueError(
                f"LinT requires 'input_feature_dim' to be an int, "
                f"got {type(config.input_feature_dim).__name__}. "
                "For multiple modalities use LinMulT."
            )

        self.name = config.name

        attention_config = config.build_attention_config()

        # 1. Projection
        self.projection = ProjectionModule(
            input_feature_dims=[config.input_feature_dim],
            d_model=config.d_model,
            dropout=config.dropout_input,
            special_handling=config.special_handling,
        )

        # 2. Self-attention transformer
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.cmt_num_layers,
            attention_config=attention_config,
            dropout_pe=config.dropout_pe,
            dropout_ffn=config.dropout_ffn,
        )

        # 3. Optional: time reduce module
        self.trm: TRM | None = None
        if config.time_dim_reducer:
            self.trm = TRM(d_model=config.d_model, reducer=config.time_dim_reducer)

        # 4. Optional: FFN fusion
        self.ffn: FFNResidual | None = None
        if config.add_module_ffn_fusion:
            self.ffn = FFNResidual(dim=config.d_model, dropout=config.dropout_output)

        # 5. Output heads
        self.output_heads = HeadModule(
            input_dim=config.d_model,
            head_configs=config.heads,
        )

    def extra_repr(self) -> str:
        """Return the model name for identification in repr output."""
        return f"name={self.name!r}"  # pragma: no cover

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        name: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the forward pass.

        Args:
            x (torch.Tensor): Input of shape ``(B, T, F)``. May also be a
                single-element list ``[tensor]``.
            mask (torch.Tensor, optional): Bool mask of shape ``(B, T)``.
                True = valid timestep. A fully-False mask is treated as ``None``.
            name (str, optional): Key used for special-handling lookup (e.g.
                weighted-sum of layer activations). May also be a single-element list.

        Returns:
            dict[str, torch.Tensor]: Mapping from head name to output tensor.
                Shape is ``(B, output_dim)`` when ``time_dim_reducer`` is set,
                otherwise ``(B, T, output_dim)``.
        """
        if isinstance(x, list):
            if len(x) == 1:
                x = x[0]
            else:
                raise ValueError(f"A single tensor is expected, got {len(x)}.")

        if isinstance(name, list):
            if len(name) == 1:
                name = name[0]
            else:
                raise ValueError(f"A single name is expected, got {len(name)}.")

        if isinstance(mask, list):
            if len(mask) == 1:
                mask = mask[0]
            else:
                raise ValueError(f"A single mask is expected, got {len(mask)}.")

        if mask is not None and not mask.any():
            mask = None

        projected = self.projection([x], names=[name] if name else None)[0]

        x = self.encoder(projected, query_mask=mask)

        if self.trm is not None:
            x = self.trm(x, mask)
            mask = None

        if self.ffn is not None:
            x = self.ffn(x)

        return self.output_heads(x, mask=mask)
