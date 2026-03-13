"""Fusion-stage module for LinMulT.

FusionModule groups all post-branch fusion steps into a single nn.Module:
  - Optional per-branch temporal reduction (TRM)
  - Optional TAM-based cross-branch alignment and fusion
  - Optional self-attention over the fused representation (SAT)
  - Optional feed-forward layer (FFN)
"""

from typing import cast

import torch
from torch import Tensor, nn

from linmult.core.attention import AttentionConfig
from linmult.core.ffn import FFNResidual
from linmult.core.temporal import TAM, TRM
from linmult.core.transformer import TransformerEncoder


def _combine_masks(x_list: list[Tensor], mask_list: list[Tensor | None]) -> Tensor | None:
    """AND-combine per-branch masks into a single joint mask.

    Returns ``None`` when all masks are ``None`` (no masking needed).
    """
    if all(m is None for m in mask_list):
        return None
    effective = [
        m if m is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for x, m in zip(x_list, mask_list)
    ]
    return torch.stack(effective, dim=0).all(dim=0)


class FusionModule(nn.Module):
    """Fuses multi-branch representations into a single representation.

    Builds all internal sub-modules (TRM, TAM, SAT, FFN) from primitive
    parameters.  Two primary fusion paths (mutually exclusive):

    - **TAM path** (when ``add_tam_fusion=True``): the TAM receives all branch
      tensors, aligns them temporally, and projects to a common dimension.
    - **Concat path**: each branch is optionally reduced along the time axis
      with TRM (when ``time_dim_reducer`` is set), then all branches are
      concatenated along the feature dimension.

    Either path is followed by an optional self-attention transformer (SAT)
    and an optional feed-forward residual layer (FFN).

    Args:
        input_dim: Per-branch feature dimension (from ``CrossModalModule.output_dim``).
        n_branches: Number of branches to fuse.
        d_model: Base model dimension (used for TAM output target).
        num_heads: Number of attention heads for internal transformers.
        attention_config: Attention configuration for internal transformers.
        time_dim_reducer: Temporal reducer type (``"attentionpool"``, ``"gap"``,
            ``"gmp"``, ``"last"``).  ``None`` to skip.
        add_tam_fusion: Whether to use TAM-based fusion.
        tam_aligner: Temporal alignment method for TAM fusion.
        tam_time_dim: Target time dimension for TAM fusion.
        fusion_num_layers: Depth of the TAM fusion transformer.
        add_sat_fusion: Whether to add a self-attention transformer after fusion.
        fusion_sat_num_layers: Depth of the fusion SAT.
        add_ffn_fusion: Whether to add a feed-forward residual layer.
        dropout_tam: Dropout for TAM fusion.
        dropout_output: Dropout for FFN.
        dropout_pe: Positional encoding dropout for internal transformers.
        dropout_ffn: FFN dropout within transformers.
    """

    def __init__(
        self,
        input_dim: int,
        n_branches: int,
        d_model: int,
        num_heads: int = 8,
        attention_config: AttentionConfig | None = None,
        *,
        time_dim_reducer: str | None = None,
        add_tam_fusion: bool = False,
        tam_aligner: str = "aap",
        tam_time_dim: int = 0,
        fusion_num_layers: int = 6,
        dropout_tam: float = 0.1,
        add_sat_fusion: bool = False,
        fusion_sat_num_layers: int = 6,
        add_ffn_fusion: bool = False,
        dropout_output: float = 0.0,
        dropout_pe: float = 0.0,
        dropout_ffn: float = 0.1,
    ) -> None:
        super().__init__()
        fusion_dim = input_dim * n_branches

        self.trm: TRM | None = None
        if time_dim_reducer:
            self.trm = TRM(d_model=input_dim, reducer=time_dim_reducer)

        self.tam: TAM | None = None
        if add_tam_fusion:
            self.tam = TAM(
                input_dim=fusion_dim,
                output_dim=n_branches * d_model,
                aligner=tam_aligner,
                time_dim=tam_time_dim,
                dropout_out=dropout_tam,
                num_layers=fusion_num_layers,
                num_heads=num_heads,
                attention_config=attention_config,
                dropout_pe=dropout_pe,
                dropout_ffn=dropout_ffn,
                name="TAM fusion",
            )
            fusion_dim = n_branches * d_model

        self.sat: TransformerEncoder | None = None
        if add_sat_fusion:
            self.sat = TransformerEncoder(
                d_model=fusion_dim,
                num_heads=num_heads,
                num_layers=fusion_sat_num_layers,
                attention_config=attention_config,
                dropout_pe=dropout_pe,
                dropout_ffn=dropout_ffn,
                name="Fusion SAT",
            )

        self.ffn: FFNResidual | None = None
        if add_ffn_fusion:
            self.ffn = FFNResidual(dim=fusion_dim, dropout=dropout_output)

        self._output_dim = fusion_dim

    @property
    def output_dim(self) -> int:
        """Final fused dimension after all fusion stages."""
        return self._output_dim

    def forward(
        self,
        x_list: list[Tensor],
        mask_list: list[Tensor | None],
    ) -> tuple[Tensor, Tensor | None]:
        """Fuse branch representations into one tensor.

        Args:
            x_list: One tensor per branch, each ``(B, T, input_dim)``.
            mask_list: Boolean mask per branch, each ``(B, T)`` or ``None``.

        Returns:
            Tuple of:
            - Fused tensor ``(B, [T,] output_dim)``.  Time axis is present when
              the TAM path is used or when TRM is not applied.
            - Joint boolean mask ``(B, T)`` or ``None`` when all masks are ``None``
              (or after temporal reduction which removes the time axis).
        """
        mask: Tensor | None

        if self.tam is not None:
            x, mask = self.tam(x_list, mask_list)
        else:
            if self.trm is not None:
                x_list = self.trm.apply_to_list(x_list, mask_list)  # (B, F) each
                mask_list = cast("list[Tensor | None]", [None] * len(mask_list))

            mask = _combine_masks(x_list, mask_list)
            x = torch.cat(x_list, dim=-1)  # (B, T, combined) or (B, combined)

        if self.sat is not None:
            x = self.sat(x, query_mask=mask)

        if self.ffn is not None:
            x = self.ffn(x)

        return x, mask
