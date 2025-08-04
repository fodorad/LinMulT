import logging
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder
from linmult.models.modules import TAM, TRM
from linmult.models.heads import HeadFactory
from linmult.models.utils import load_config


class LinMulT(nn.Module):

    def __init__(self, config: dict | str):
        """Construct LinMulT: Linear-complexity Multimodal Transformer."""
        super().__init__()

        if isinstance(config, str):
            config = load_config(config)

        self.input_feature_dim = config.get("input_feature_dim")
        self.special_handling = config.get("special_handling", {})
        self.n_sequences = len(self.input_feature_dim)
        self.d_model = config.get("d_model", 40)
        self.dropout_input = config.get("dropout_input", 0.)
        self.dropout_output = config.get("dropout_output", 0.)
        self.module_time_dim_reducer = config.get("time_dim_reducer", None)
        self.module_multimodal_signal = config.get("multimodal_signal", None)
        self.module_tam_fusion = config.get("tam_fusion", None)
        self.module_self_attention_fusion = config.get("sa_fusion", None)
        self.module_ffn_fusion = config.get("ffn_fusion", None)
        self.module_unimodal_sat = config.get("unimodal_sat", None)
        self.head_configs = config.get("heads", [])
        self.auxiliary_head_configs = config.get("auxiliary_heads", [])

        # Initialize stages
        self._init_projections()
        self._init_unimodal_transformers(config)
        self._init_module_multimodal_signal(config)
        self._init_crossmodal_transformers(config)
        self._init_auxiliary_heads()
        self._init_module_time_dim_reducer(config)
        self._init_module_fusion(config)
        self._init_output_heads()


    def forward(self,
            inputs: list[torch.Tensor],
            masks: list[torch.BoolTensor] | None = None,
            names: list[str] | None = None
        ) -> list[torch.Tensor]:
        """Inference with LinMulT.

        Args:
            inputs (list[torch.Tensor]): Input tensors of shape (B, T, F).
            masks (list[torch.BoolTensor], Optional): Masks for each sequence. Defaults to None.
            names (list[str], Optional): Feature names for special aggregation. Defaults to None.

        Returns:
            (list[torch.Tensor]): tensor of shape (B, F) and/or (B, T, F) based on output head definition
        """
        if masks is None:
            masks = [None] * self.n_sequences
        else:
            masks = [mask if mask is not None and mask.any() else None for mask in masks] # full masks are changed to None

        logging.debug(f'input sizes: {[tuple(x.shape) for x in inputs]}')
        projected_inputs = self._apply_projections(inputs, names)
        logging.debug(f'projected input sizes: {[tuple(x.shape) for x in projected_inputs]}')
        projected_inputs, masks = self._apply_multimodal_signal(projected_inputs, masks)
        logging.debug(f'projected input sizes (with multimodal signal): {[tuple(x.shape) for x in projected_inputs]}')
        branch_representations = self._apply_branch(projected_inputs, masks)
        logging.debug(f'branch representation sizes: {[tuple(x.shape) for x in branch_representations]}')
        outputs_aux = self._apply_auxiliary_heads(branch_representations, masks)
        fused_representation, mask = self._apply_fusion(branch_representations, masks)
        logging.debug(f'fused representation size: {tuple(fused_representation.shape)}')
        outputs = self._apply_output_heads(fused_representation, mask)
        logging.debug(f'output sizes: {"".join([f"{name}: {tuple(x.shape)}" for name, x in outputs.items()])}')
        
        if self.auxiliary_head_configs:
            return outputs, outputs_aux
        return outputs


    def _init_projections(self):
        """Initialize the input-specific temporal projection layers."""
        self.projectors = nn.ModuleList([
            nn.Conv1d(input_channels, self.d_model, kernel_size=1, padding=0, bias=False)
            for input_channels in self.input_feature_dim
        ])

        self.special_modules = {}
        for name, params in self.special_handling.items():
            if params['type'] == "weighted_sum":
                start_layer = params['start_layer']
                end_layer = params['end_layer']
                n_layers = end_layer - start_layer
                self.special_modules[name] = nn.Parameter(
                    torch.ones(n_layers) / n_layers, requires_grad=True
                )


    def _init_module_time_dim_reducer(self, config: dict):
        """Initialize a Time Reducer Module"""
        if self.module_time_dim_reducer:
            self.trm = TRM(config=config)


    def _init_module_multimodal_signal(self, config: dict):
        """Initialize a Time Aligner Module to create a multimodal signal"""
        if self.module_multimodal_signal:
            self.tam_mms = TAM(config=config.copy() | {
                "name": "TAM MMS",
                "src_dim": self.n_sequences * self.d_model,
                "tgt_dim": self.d_model,
                "n_layers": config.get("n_layers_mms", 6)
            })


    def _init_unimodal_transformers(self, config: dict):
        if self.module_unimodal_sat:
            self.unimodal_transformers = nn.ModuleList([
                TransformerEncoder(config=config.copy() | {"name": f"Unimodal SA {input_index}"})
                for input_index in range(self.n_sequences)
            ])


    def _init_crossmodal_transformers(self, config: dict):
        """Initialize input-branches: cross-modal and self-attention transformers."""        
        self.branch_crossmodal_transformers = nn.ModuleList()
        for target_index in range(self.n_sequences):
            input_indices = [i for i in range(self.n_sequences) if i != target_index]
            crossmodal_transformers = [
                TransformerEncoder(config=config.copy() | {"name": f"CMT {input_index}->{target_index}"})
                for input_index in input_indices
            ]

            if self.module_multimodal_signal:
                crossmodal_transformers.append(
                    TransformerEncoder(config=config.copy() | {"name": f"CMT mms->{target_index}"})
                )

            self.branch_crossmodal_transformers.append(nn.ModuleList(crossmodal_transformers))

        combined_dim = ((self.n_sequences) if self.module_multimodal_signal else (self.n_sequences - 1)) * self.d_model
        self.branch_self_attention_transformers = nn.ModuleList([
            TransformerEncoder(
                config=config.copy() | {
                    "name": f"SAT {target_index}",
                    "d_model": combined_dim,
                    "n_layers": config.get("n_layers_sa", 6),
                }
            )
            for target_index in range(self.n_sequences)
        ])


    def _init_module_fusion(self, config: dict):
        """Initialize fusion mechanisms."""
        combined_dim = ((self.n_sequences) if self.module_multimodal_signal else self.n_sequences-1) * self.n_sequences * self.d_model
        if self.module_unimodal_sat: combined_dim += self.d_model * self.n_sequences

        if self.module_tam_fusion:

            self.tam_fusion = TAM(config=config.copy() | {
                "name": "TAM fusion",
                "src_dim": combined_dim,
                "tgt_dim": self.n_sequences * self.d_model,
                "n_layers": config.get("n_layers_fusion", 6)
            })

        if self.module_self_attention_fusion:
            self.self_attention_fusion_transformer = TransformerEncoder(
                config=config.copy() | {
                    "name": "Fusion SA",
                    "d_model": combined_dim,
                    "n_layers": config.get("n_layers_sa_fusion", 6),
                }
            )

        if self.module_ffn_fusion:
            self.projection_1 = nn.Linear(combined_dim, combined_dim)
            self.projection_2 = nn.Linear(combined_dim, combined_dim)


    def _init_output_heads(self):
        """Initialize output layers."""
        n_cmt = self.n_sequences if self.module_multimodal_signal else self.n_sequences - 1
        n_sat = self.n_sequences
        fusion_dim_multiplier = 1 / self.n_sequences if self.module_tam_fusion else 1
        combined_dim = max(int(n_cmt * n_sat * fusion_dim_multiplier), 2) * self.d_model

        self.output_heads = nn.ModuleDict()
        for i, head_cfg in enumerate(self.head_configs):
            head = HeadFactory.create_head(
                type=head_cfg['type'],
                input_dim=combined_dim,
                output_dim=head_cfg['output_dim'],
                config=head_cfg
            )
            head_name = head_cfg.get('name', f'head_{i}')
            self.output_heads[head_name] = head


    def _init_auxiliary_heads(self):
        """Initialize auxiliary output layers."""
        if not self.auxiliary_head_configs:
            return

        combined_dim = ((self.n_sequences) if self.module_multimodal_signal else (self.n_sequences - 1)) * self.d_model
        if self.module_unimodal_sat: combined_dim += self.d_model

        self.auxiliary_heads = nn.ModuleList()
        for input_index in range(self.n_sequences):
            input_aux_dict = nn.ModuleDict()
            for head_index, head_cfg in enumerate(self.auxiliary_head_configs):
                aux_head = HeadFactory.create_head(
                    type=head_cfg['type'],
                    input_dim=combined_dim,
                    output_dim=head_cfg['output_dim'],
                    config=head_cfg
                )
                aux_head_name = head_cfg.get('name', f'aux_head_{head_index}') + f'_{input_index}'
                input_aux_dict[aux_head_name] = aux_head
            self.auxiliary_heads.append(input_aux_dict)


    def _apply_projections(self,
            x_list: list[torch.Tensor],
            names: list[str] | None = None
        ) -> list[torch.Tensor]:
        """Apply temporal convolution projection and special handling for specified sequences.

        Args:
            x_list (list[torch.Tensor]): List of input tensors [(B, T, F), ..., (B, N, T, F)].
            names (list[str], Optional): List of input names corresponding to x_list. Defaults to None.

        Returns:
            list[torch.Tensor]: List of projected tensors [(B, T, d_model), ...].
        """
        if names is None: names = [None] * self.n_sequences

        projected_list = []
        for projector, x, name in zip(self.projectors, x_list, names):
            if name in self.special_handling:
                params = self.special_handling[name]
                if params['type'] == "weighted_sum" and x.ndim == 4:
                    # Handle (B, N, T, F) tensors with weighted aggregation
                    x = x[:,params['start_layer']:,:,:]
                    weights = F.softmax(self.special_modules[name], dim=0)  # Normalize weights
                    weights = weights.to(x.device)
                    x = torch.einsum("n,bntf->btf", weights, x)  # Weighted sum: (B, N, T, F) -> (B, T, F)

            # Apply temporal projection
            projected_x = projector(
                F.dropout(
                    x.transpose(1, 2),  # (B, T, F) -> (B, F, T)
                    p=self.dropout_input,
                    training=self.training
                )
            ).transpose(1, 2)  # (B, F, T) -> (B, T, d_model)
            projected_list.append(projected_x)

        return projected_list


    def _apply_multimodal_signal(self,
        x_list: list[torch.Tensor],
        mask_list: list[torch.BoolTensor | None]
    ) -> list[torch.Tensor]:
        """Create a multimodal signal and add them to the input list"""
        if self.module_multimodal_signal:
            mm_x, mm_mask = self.tam_mms(x_list, mask_list)
            x_list.append(mm_x)
            mask_list.append(mm_mask)
        return x_list, mask_list


    def _apply_branch(self,
            x_list: list[torch.Tensor],
            mask_list: list[torch.BoolTensor | None]
        ) -> list[torch.Tensor]:
        """Perform input-branch: cross-modal and self-attention transformers."""
        branch_representations = []
        for target_index in range(self.n_sequences):
            input_indices = [i for i in range(self.n_sequences) if i != target_index]
            cross_modal_hidden = torch.cat([
                    self.branch_crossmodal_transformers[target_index][i](
                        x_list[target_index],
                        x_list[input_index],
                        x_list[input_index],
                        query_mask=mask_list[target_index],
                        key_mask=mask_list[input_index],
                    )
                    for i, input_index in enumerate(input_indices)
                ] + ([self.branch_crossmodal_transformers[target_index][self.n_sequences-1](
                            x_list[target_index],
                            x_list[self.n_sequences],
                            x_list[self.n_sequences],
                            query_mask=mask_list[target_index],
                            key_mask=mask_list[self.n_sequences],
                    )] if self.module_multimodal_signal else []),
                dim=2
            )

            sat_representation = self.branch_self_attention_transformers[target_index](cross_modal_hidden, query_mask=mask_list[target_index])

            if self.module_unimodal_sat:
                branch_representation = torch.cat([
                    sat_representation,
                    self.unimodal_transformers[target_index](x_list[target_index], query_mask=mask_list[target_index])
                ], dim=2)
            else:
                branch_representation = sat_representation

            branch_representations.append(branch_representation)

        return branch_representations


    def _apply_fusion(self, x_list: list[torch.Tensor], mask_list: list[torch.BoolTensor | None]) -> torch.Tensor:
        """Apply fusion mechanisms."""
        if self.module_tam_fusion:
            x, mask = self.tam_fusion(x_list, mask_list)
        else:
            if self.module_time_dim_reducer:
                # aggregated over the time dim
                x_list = self.trm.apply_to_list(x_list, mask_list) # (B, F)
                mask_list = [None] * len(mask_list)

            x = torch.cat(x_list, dim=-1) # [(B, T, F), ...] -> (B, T, combined_dim) or [(B, F), ...] -> (B, combined_dim)  
            mask = torch.stack(mask_list, dim=0).any(dim=0) if all(mask is not None for mask in mask_list) else None # timestep-wise logical OR; [(B, F), ...] -> (B, T)

        if self.module_self_attention_fusion:
            x = self.self_attention_fusion_transformer(x, query_mask=mask) # (B, T, F*N) and (B, T)

        if self.module_ffn_fusion:
            x = self.projection_2(
                F.dropout(
                    F.gelu(self.projection_1(x)),
                    p=self.dropout_output,
                    training=self.training
                )
            ) + x # ffn + residual

        return x, mask


    def _apply_auxiliary_heads(self, x_list: list[torch.Tensor], mask_list: list[torch.BoolTensor | None]) -> list[dict[str, torch.Tensor]]:
        """Apply auxiliary heads"""
        if not self.auxiliary_head_configs:
            return None

        return [
            {name: head(x, mask=mask) for name, head in aux_heads.items()} 
            for x, mask, aux_heads 
            in zip(x_list, mask_list, self.auxiliary_heads)
        ]


    def _apply_output_heads(self, x: torch.Tensor, mask: torch.BoolTensor | None) -> dict[str, torch.Tensor]:
        """Apply output heads"""
        return {name: head(x, mask=mask) for name, head in self.output_heads.items()}


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config('configs/LinMulT_with_aux.yaml')
    model = LinMulT(config)

    x_1 = torch.rand((8, 300, 25))
    x_2 = torch.rand((8, 300, 41))
    x_3 = torch.rand((8, 500, 768))
    m_1 = torch.ones((8, 300), dtype=bool)
    m_2 = torch.ones((8, 300), dtype=bool)
    m_3 = torch.zeros((8, 500), dtype=bool)
    outputs, outputs_aux = model([x_1, x_2, x_3], [m_1, m_2, m_3])
    for i, (name, out) in enumerate(outputs.items()):
        print(f"Head {i+1} ({name}) output shape: {out.shape}")

    for i, aux_head_dict in enumerate(outputs_aux):
        for name, out in aux_head_dict.items():
            print(f"Auxiliary head {i+1} ({name}) output shape: {out.shape}")
