"""Projection module for LinMulT and LinT.

ProjectionModule handles per-modality feature projection with optional
special handling (e.g. weighted-sum over transformer layers) and optional
per-modality TCN for temporal smoothing.
"""

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from linmult.core.tcn import TCN


class ProjectionModule(nn.Module):
    """Projects each modality's features to a shared dimension.

    For each modality, applies a 1-D convolution (kernel=1) that maps
    ``input_feature_dims[i]`` to ``d_model``.  Before projection, optional
    special handling (e.g. weighted-sum aggregation over stacked transformer
    layers) can reduce a 4-D input ``(B, N, T, F)`` to ``(B, T, F)``.

    When TCN is enabled, a per-modality :class:`~linmult.core.tcn.TCN`
    is applied after projection to smooth frame-level features temporally.

    Args:
        input_feature_dims (list[int]): Feature dimension per modality.
        d_model (int): Target projection dimension.
        dropout (float): Dropout applied to inputs before projection.
        special_handling (dict[str, Any], optional): Dict mapping modality names to
            handling specs. Currently supports
            ``{"type": "weighted_sum", "start_layer": int, "end_layer": int}``.
        add_tcn (bool): Enable per-modality TCN after projection.
            Defaults to ``True``.
        tcn_num_layers (int): Number of dilated causal convolution layers.
            Defaults to ``3``.
        tcn_kernel_size (int): Kernel size for each TCN layer.  Defaults to ``3``.
        tcn_dropout (float): Dropout in each TCN layer.  Defaults to ``0.1``.
    """

    def __init__(
        self,
        input_feature_dims: list[int],
        d_model: int,
        dropout: float = 0.0,
        special_handling: dict[str, Any] | None = None,
        add_tcn: bool = True,
        tcn_num_layers: int = 3,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.special_handling: dict[str, Any] = special_handling or {}

        self.projectors = nn.ModuleList(
            [
                nn.Conv1d(in_ch, d_model, kernel_size=1, padding=0, bias=False)
                for in_ch in input_feature_dims
            ]
        )

        self.tcn_blocks: nn.ModuleList | None = None
        if add_tcn:
            self.tcn_blocks = nn.ModuleList(
                [
                    TCN(d_model, tcn_num_layers, tcn_kernel_size, tcn_dropout)
                    for _ in input_feature_dims
                ]
            )

        self.special_modules = nn.ParameterDict()
        for name, params in self.special_handling.items():
            if params["type"] == "weighted_sum":
                n_layers = params["end_layer"] - params["start_layer"]
                self.special_modules[name] = nn.Parameter(
                    torch.ones(n_layers) / n_layers, requires_grad=True
                )

    def forward(
        self,
        x_list: list[Tensor],
        names: list[str] | None = None,
    ) -> list[Tensor]:
        """Project each modality to ``d_model``.

        Args:
            x_list: One tensor per modality, each ``(B, T, F)`` or
                ``(B, N, T, F)`` for weighted-sum inputs.
            names: Optional modality names for special handling lookup.

        Returns:
            List of projected tensors, each ``(B, T, d_model)``.
        """
        names_list: list[str | None] = (
            list(names) if names is not None else cast("list[str | None]", [None] * len(x_list))
        )

        projected_list = []
        for projector, x, name in zip(self.projectors, x_list, names_list):
            if name in self.special_handling:
                assert isinstance(name, str)
                params = self.special_handling[name]
                if params["type"] == "weighted_sum" and x.ndim == 4:
                    x = x[:, params["start_layer"] : params["end_layer"], :, :]
                    weights = F.softmax(self.special_modules[name], dim=0)
                    weights = weights.to(x.device)
                    x = torch.einsum("n,bntf->btf", weights, x)

            projected_x = projector(
                F.dropout(x.transpose(1, 2), p=self.dropout, training=self.training)
            ).transpose(1, 2)
            projected_list.append(projected_x)

        if self.tcn_blocks is not None:
            projected_list = [tcn(proj) for tcn, proj in zip(self.tcn_blocks, projected_list)]

        return projected_list
