"""LinMulT: multimodal linear-complexity transformer."""

import logging
from pathlib import Path
from typing import cast

import torch
from torch import nn

from linmult.core.branch import CrossModalModule
from linmult.core.config import HeadConfig, LinMulTConfig
from linmult.core.fusion import FusionModule
from linmult.core.heads import HeadFactory, HeadModule
from linmult.core.projection import ProjectionModule


class LinMulT(nn.Module):
    """Multimodal Linear-complexity Transformer.

    Processes 2 or more time-series modalities through a cross-modal attention
    pipeline and applies configurable output heads.

    Args:
        config (LinMulTConfig | str | Path): Configuration object or path to a YAML
            file.  When a path is given the file is loaded with
            :meth:`LinMulTConfig.from_yaml`.
    """

    def __init__(self, config: LinMulTConfig | str | Path):
        """Initialize LinMulT.

        Args:
            config (LinMulTConfig | str | Path): Configuration object or path to a YAML
                file. When a path is given the file is loaded with
                :meth:`LinMulTConfig.from_yaml`.

        Raises:
            ValueError: If fewer than 2 input modalities are provided.
        """
        super().__init__()

        if isinstance(config, str | Path):
            config = LinMulTConfig.from_yaml(config)

        if not isinstance(config.input_feature_dim, list) or len(config.input_feature_dim) < 2:
            raise ValueError(
                f"LinMulT requires at least 2 input modalities, "
                f"got {config.input_feature_dim!r}. "
                "For a single-modality model use LinT or a standard TransformerEncoder directly."
            )

        self.name = config.name
        self.n_modalities = len(config.input_feature_dim)
        self.auxiliary_head_configs = config.auxiliary_heads

        attention_config = config.build_attention_config()

        # --- Pipeline modules ---

        self.projection = ProjectionModule(
            input_feature_dims=config.input_feature_dim,
            d_model=config.d_model,
            dropout=config.dropout_input,
            special_handling=config.special_handling,
        )

        self.cross_modal = CrossModalModule(
            num_modalities=self.n_modalities,
            d_model=config.d_model,
            num_heads=config.num_heads,
            branch_cmt_num_layers=config.cmt_num_layers,
            branch_sat_num_layers=config.branch_sat_num_layers,
            attention_config=attention_config,
            add_mms=config.add_module_multimodal_signal,
            mms_num_layers=config.mms_num_layers,
            tam_aligner=config.tam_aligner or "aap",
            tam_time_dim=config.tam_time_dim or 0,
            dropout_tam=config.dropout_tam,
            add_unimodal_sat=config.add_module_unimodal_sat,
            unimodal_sat_num_layers=config.unimodal_sat_num_layers,
            dropout_pe=config.dropout_pe,
            dropout_ffn=config.dropout_ffn,
        )

        self.fusion = FusionModule(
            input_dim=self.cross_modal.output_dim,
            n_branches=self.n_modalities,
            d_model=config.d_model,
            num_heads=config.num_heads,
            attention_config=attention_config,
            time_dim_reducer=config.time_dim_reducer,
            add_tam_fusion=config.add_module_tam_fusion,
            tam_aligner=config.tam_aligner or "aap",
            tam_time_dim=config.tam_time_dim or 0,
            fusion_num_layers=config.fusion_num_layers,
            dropout_tam=config.dropout_tam,
            add_sat_fusion=config.add_module_sat_fusion,
            fusion_sat_num_layers=config.fusion_sat_num_layers,
            add_ffn_fusion=config.add_module_ffn_fusion,
            dropout_output=config.dropout_output,
            dropout_pe=config.dropout_pe,
            dropout_ffn=config.dropout_ffn,
        )

        self.output_heads = HeadModule(
            input_dim=self.fusion.output_dim,
            head_configs=config.heads,
        )

        self.auxiliary_heads: nn.ModuleList | None = None
        if config.auxiliary_heads:
            self.auxiliary_heads = _build_auxiliary_heads(
                n_branches=self.n_modalities,
                input_dim=self.cross_modal.output_dim,
                head_configs=config.auxiliary_heads,
            )

    def extra_repr(self) -> str:
        """Return the model name for identification in repr output."""
        return f"name={self.name!r}"  # pragma: no cover

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: list[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
        names: list[str] | None = None,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]] | None]
    ):
        """Run the full LinMulT forward pass.

        Args:
            inputs (list[torch.Tensor]): One tensor per modality, each of shape
                ``(B, T, F)`` or ``(B, N, T, F)`` for special weighted-sum inputs.
            masks (list[torch.Tensor], optional): Boolean mask per modality, each
                of shape ``(B, T)``. ``True`` = valid timestep. ``None`` entries and
                fully-valid masks are treated as no mask. Defaults to ``None``.
            names (list[str], optional): Feature names for special aggregation
                (e.g. weighted-sum over transformer layers). Defaults to ``None``.

        Returns:
            dict[str, torch.Tensor]: Output dict mapping head names to predictions.
                Each value has shape ``(B, output_dim)`` or ``(B, T, output_dim)``
                depending on the head type.
            tuple[dict, list[dict]]: When ``auxiliary_heads`` are configured, a tuple
                where the first element is the primary output dict and the second
                element is a list of per-branch auxiliary prediction dicts.

        Raises:
            ValueError: If the number of input tensors does not match ``n_modalities``.
        """
        if len(inputs) != self.n_modalities:
            raise ValueError(f"Expected {self.n_modalities} input tensors, got {len(inputs)}.")

        mask_list = _normalize_masks(masks, self.n_modalities)

        logging.debug(f"input sizes: {[tuple(x.shape) for x in inputs]}")
        projected = self.projection(inputs, names)
        logging.debug(f"projected input sizes: {[tuple(x.shape) for x in projected]}")

        branch_reps = self.cross_modal(projected, mask_list)
        logging.debug(f"branch representation sizes: {[tuple(x.shape) for x in branch_reps]}")

        outputs_aux = _apply_auxiliary_heads(self.auxiliary_heads, branch_reps, mask_list)
        fused, mask = self.fusion(branch_reps, mask_list)
        logging.debug(f"fused representation size: {tuple(fused.shape)}")

        outputs = self.output_heads(fused, mask=mask)
        logging.debug(
            f"output sizes: {''.join([f'{n}: {tuple(x.shape)}' for n, x in outputs.items()])}"
        )

        if self.auxiliary_head_configs:
            return outputs, outputs_aux
        return outputs


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _normalize_masks(
    masks: list[torch.Tensor] | None, n_modalities: int
) -> list[torch.Tensor | None]:
    """Convert user-provided masks to internal format.

    Fully-valid masks (all True) are replaced with ``None`` for efficiency.
    """
    if masks is None:
        return cast("list[torch.Tensor | None]", [None] * n_modalities)
    return [mask if mask is not None and not mask.all() else None for mask in masks]


def _build_auxiliary_heads(
    n_branches: int,
    input_dim: int,
    head_configs: list[HeadConfig],
) -> nn.ModuleList:
    """Build per-branch auxiliary heads."""
    aux_list = nn.ModuleList()
    for i in range(n_branches):
        aux_dict = nn.ModuleDict()
        for j, cfg in enumerate(head_configs):
            head = HeadFactory.create_head(
                type=cfg.type,
                input_dim=input_dim,
                output_dim=cfg.output_dim,
                config=cfg,
            )
            prefix = cfg.name if cfg.name else f"aux_head_{j}"
            aux_dict[f"{prefix}_{i}"] = head
        aux_list.append(aux_dict)
    return aux_list


def _apply_auxiliary_heads(
    auxiliary_heads: nn.ModuleList | None,
    x_list: list[torch.Tensor],
    mask_list: list[torch.Tensor | None],
) -> list[dict[str, torch.Tensor]] | None:
    """Apply auxiliary heads to each branch representation."""
    if auxiliary_heads is None:
        return None
    return [
        {name: head(x, mask=mask) for name, head in aux_heads.items()}
        for x, mask, aux_heads in zip(x_list, mask_list, auxiliary_heads)
    ]
