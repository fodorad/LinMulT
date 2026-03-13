"""Branch modules for LinMulT.

- MultimodalSignal: creates a shared multimodal token from all modality sequences
- CrossModalBranch: processes one target modality through cross-modal attention from all sources
- CrossModalModule: orchestrates all branches with optional MMS
"""

from typing import cast

import torch
from torch import Tensor, nn

from linmult.core.attention import AttentionConfig
from linmult.core.temporal import TAM
from linmult.core.transformer import TransformerEncoder


class MultimodalSignal(nn.Module):
    """Creates a shared multimodal signal from all modality sequences.

    Wraps a TAM that fuses all modality sequences into a single aligned representation.
    The output is appended to ``x_list`` / ``mask_list`` so that each branch can
    attend to it via its final cross-modal transformer.

    Args:
        tam (TAM): Temporal alignment module that receives all modality sequences.
    """

    def __init__(self, tam: TAM) -> None:
        super().__init__()
        self.tam = tam

    def forward(
        self,
        x_list: list[Tensor],
        mask_list: list[Tensor | None],
    ) -> tuple[list[Tensor], list[Tensor | None]]:
        """Compute multimodal signal and append to input lists.

        Args:
            x_list: One tensor per modality, each ``(B, T_i, d_model)``.
            mask_list: Boolean mask per modality, each ``(B, T_i)`` or ``None``.

        Returns:
            Extended ``(x_list, mask_list)`` with the multimodal signal appended as
            the last element.  The signal has shape ``(B, time_dim, tgt_dim)``.
        """
        mms_x, mms_mask = self.tam(x_list, mask_list)
        return list(x_list) + [mms_x], list(mask_list) + [mms_mask]


class CrossModalBranch(nn.Module):
    """Processes one target modality through cross-modal and self-attention.

    Forward pass:
      1. Run one cross-modal TransformerEncoder per source (query=target, key/value=source).
      2. Concatenate all cross-modal outputs → ``(B, T, branch_dim)``.
      3. Apply branch self-attention (SAT).
      4. Optionally concatenate with unimodal SAT output.

    Args:
        cross_transformers: One cross-modal TransformerEncoder per source modality
            (including the MMS token if enabled).
        branch_sat: Self-attention TransformerEncoder applied over the concatenated
            cross-modal representation.
        unimodal_sat: Optional self-attention TransformerEncoder applied to the
            original (projected) query sequence.  Its output is concatenated with
            the SAT output before returning.
    """

    def __init__(
        self,
        cross_transformers: nn.ModuleList,
        branch_sat: TransformerEncoder,
        unimodal_sat: TransformerEncoder | None = None,
    ) -> None:
        super().__init__()
        self.cross_transformers = cross_transformers
        self.branch_sat = branch_sat
        self.unimodal_sat = unimodal_sat

    def forward(
        self,
        x_query: Tensor,
        x_sources: list[Tensor],
        mask_query: Tensor | None = None,
        mask_sources: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Run cross-modal + self-attention for one target modality.

        Args:
            x_query: Target modality tensor ``(B, T_q, d_model)``.
            x_sources: Source modality tensors, each ``(B, T_s, d_model)``.
                Must be in the same order as ``cross_transformers``.
            mask_query: Boolean mask ``(B, T_q)`` for the query. ``None`` = no mask.
            mask_sources: Boolean masks for each source, same length as ``x_sources``.
                ``None`` entries treated as no mask; omit the list to use no masks.

        Returns:
            Branch representation ``(B, T_q, full_branch_dim)`` where
            ``full_branch_dim = len(x_sources) * d_model [+ d_model if unimodal_sat]``.
        """
        if mask_sources is None:
            mask_sources = cast("list[torch.Tensor | None]", [None] * len(x_sources))

        cross_outputs = [
            cmt(x_query, src, src, query_mask=mask_query, key_mask=mask_src)
            for cmt, src, mask_src in zip(self.cross_transformers, x_sources, mask_sources)
        ]
        hidden = torch.cat(cross_outputs, dim=-1)  # (B, T_q, branch_dim)
        out = self.branch_sat(hidden, query_mask=mask_query)  # (B, T_q, branch_dim)

        if self.unimodal_sat is not None:
            uni = self.unimodal_sat(x_query, query_mask=mask_query)  # (B, T_q, d_model)
            out = torch.cat([out, uni], dim=-1)

        return out


class CrossModalModule(nn.Module):
    """Orchestrates cross-modal attention across all modalities.

    Builds and manages all cross-modal branches, including optional multimodal
    signal (MMS) generation.  Each target modality gets its own
    :class:`CrossModalBranch` with cross-modal transformers from every other
    source (and from MMS if enabled).

    Args:
        num_modalities: Number of input modalities (>= 2).
        d_model: Model dimension (shared across all transformers).
        num_heads: Number of attention heads.
        branch_cmt_num_layers: Depth of each cross-modal transformer.
        branch_sat_num_layers: Depth of each branch self-attention transformer.
        attention_config: Attention configuration for all internal transformers.
        add_mms: Whether to create a multimodal signal via TAM.
        mms_num_layers: Depth of the MMS transformer (only when ``add_mms=True``).
        tam_aligner: Temporal alignment method for MMS.
        tam_time_dim: Target time dimension for MMS alignment.
        dropout_tam: Dropout for MMS TAM.
        add_unimodal_sat: Whether to add a unimodal self-attention per branch.
        unimodal_sat_num_layers: Depth of unimodal self-attention.
        dropout_pe: Positional encoding dropout.
        dropout_ffn: FFN dropout (within transformers).
    """

    def __init__(
        self,
        num_modalities: int,
        d_model: int,
        num_heads: int = 8,
        branch_cmt_num_layers: int = 6,
        branch_sat_num_layers: int = 6,
        attention_config: AttentionConfig | None = None,
        *,
        add_mms: bool = False,
        mms_num_layers: int = 6,
        tam_aligner: str = "aap",
        tam_time_dim: int = 0,
        dropout_tam: float = 0.1,
        add_unimodal_sat: bool = False,
        unimodal_sat_num_layers: int = 6,
        dropout_pe: float = 0.0,
        dropout_ffn: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.d_model = d_model
        self._add_mms = add_mms
        self._add_unimodal_sat = add_unimodal_sat

        # Shared encoder params
        self._num_heads = num_heads
        self._attention_config = attention_config
        self._dropout_pe = dropout_pe
        self._dropout_ffn = dropout_ffn

        # MMS (optional)
        self.mms: MultimodalSignal | None = None
        if add_mms:
            self.mms = MultimodalSignal(
                tam=TAM(
                    input_dim=num_modalities * d_model,
                    output_dim=d_model,
                    aligner=tam_aligner,
                    time_dim=tam_time_dim,
                    dropout_out=dropout_tam,
                    num_layers=mms_num_layers,
                    num_heads=num_heads,
                    attention_config=attention_config,
                    dropout_pe=dropout_pe,
                    dropout_ffn=dropout_ffn,
                    name="TAM MMS",
                )
            )

        # Branches — one per target modality
        n_sources = self.num_modalities if self._add_mms else self.num_modalities - 1
        branches = []
        for tgt in range(self.num_modalities):
            sources = [i for i in range(self.num_modalities) if i != tgt]
            cross_transformers = nn.ModuleList(
                [
                    self._make_encoder(
                        name=f"CMT {src}->{tgt}",
                        num_layers=branch_cmt_num_layers,
                        is_cross_modal=True,
                    )
                    for src in sources
                ]
            )
            if add_mms:
                cross_transformers.append(
                    self._make_encoder(
                        name=f"CMT mms->{tgt}",
                        num_layers=branch_cmt_num_layers,
                        is_cross_modal=True,
                    )
                )
            sat = self._make_encoder(
                name=f"SAT {tgt}",
                num_layers=branch_sat_num_layers,
                d_model=n_sources * d_model,
            )
            unimodal_sat = None
            if add_unimodal_sat:
                unimodal_sat = self._make_encoder(
                    name=f"Unimodal SAT {tgt}",
                    num_layers=unimodal_sat_num_layers,
                )
            branches.append(
                CrossModalBranch(
                    cross_transformers=cross_transformers,
                    branch_sat=sat,
                    unimodal_sat=unimodal_sat,
                )
            )
        self.branches = nn.ModuleList(branches)

    def _make_encoder(
        self,
        *,
        name: str,
        num_layers: int,
        d_model: int | None = None,
        is_cross_modal: bool = False,
    ) -> TransformerEncoder:
        """Create a TransformerEncoder with shared settings."""
        return TransformerEncoder(
            name=name,
            d_model=d_model if d_model is not None else self.d_model,
            num_heads=self._num_heads,
            num_layers=num_layers,
            attention_config=self._attention_config,
            is_cross_modal=is_cross_modal,
            dropout_pe=self._dropout_pe,
            dropout_ffn=self._dropout_ffn,
        )

    @property
    def output_dim(self) -> int:
        """Per-branch output dimension (including optional unimodal SAT)."""
        n_sources = self.num_modalities if self._add_mms else self.num_modalities - 1
        dim = n_sources * self.d_model
        if self._add_unimodal_sat:
            dim += self.d_model
        return dim

    def forward(
        self,
        x_list: list[Tensor],
        mask_list: list[Tensor | None],
    ) -> list[Tensor]:
        """Run cross-modal attention for all target modalities.

        Args:
            x_list: One projected tensor per modality, each ``(B, T_i, d_model)``.
            mask_list: Boolean mask per modality, each ``(B, T_i)`` or ``None``.

        Returns:
            List of ``num_modalities`` branch representations, each
            ``(B, T_tgt, output_dim)``.
        """
        if self.mms is not None:
            x_list, mask_list = self.mms(x_list, mask_list)

        branch_reps = []
        for tgt in range(self.num_modalities):
            src_indices = [j for j in range(self.num_modalities) if j != tgt]
            if self.mms is not None:
                src_indices.append(self.num_modalities)
            branch_reps.append(
                self.branches[tgt](
                    x_query=x_list[tgt],
                    x_sources=[x_list[j] for j in src_indices],
                    mask_query=mask_list[tgt],
                    mask_sources=[mask_list[j] for j in src_indices],
                )
            )
        return branch_reps
