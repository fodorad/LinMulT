import logging
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder, TimeReduceFactory, TemporalAlignerFactory
from linmult.models.config_loader import load_config


class LinMulT(nn.Module):

    def __init__(self, config: dict | str):
        """Construct LinMulT: Linear-complexity Multimodal Transformer."""
        super().__init__()

        if isinstance(config, str):
            config = load_config(config)

        self.input_modality_channels = config.get("input_modality_channels") # [M_1, ..., M_N]
        self.output_dim = config.get("output_dim") # [O_1, ..., O_K]
        self.d_model = config.get("d_model", 40)
        self.dropout_embedding = config.get("dropout_embedding", 0.)
        self.dropout_output = config.get("dropout_output", 0.)
        self.module_time_reduce = config.get("module_time_reduce", None)
        self.module_multimodal_signal = config.get("multimodal_signal_type", None)
        self.n_sequences = len(self.input_modality_channels) if self.module_multimodal_signal is None else len(self.input_modality_channels) + 1 # N
        
        self._validate_config()

        # Initialize stages
        self._init_projections()
        self._init_multimodal_signal_module(config)
        self._init_crossmodal_transformers(config)
        self._init_time_reduce_module(config)
        self._init_fusion_module(config)
        self._init_output_heads()


    def forward(self,
            inputs: list[torch.Tensor],
            masks: list[torch.BoolTensor] | None = None
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference with LinMulT.

        Args:
            inputs (list[torch.Tensor]): Input tensors of shape (B, T, F).
            masks (list[torch.BoolTensor] | None): Optional masks for each modality.

        Returns:
            (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): tensor of shape (B, F) and/or (B, T, F)
        """
        logging.debug(f'input sizes: {[tuple(x.shape) for x in inputs]}')
        projected_inputs = self._apply_projections(inputs)
        logging.debug(f'projected input sizes: {[tuple(x.shape) for x in projected_inputs]}')
        projected_inputs, masks = self._apply_multimodal_signal(projected_inputs, masks)
        logging.debug(f'projected input sizes (with multimodal signal): {[tuple(x.shape) for x in projected_inputs]}')
        branch_representations = self._apply_modality_branch(projected_inputs, masks)
        logging.debug(f'branch representation sizes: {[tuple(x.shape) for x in branch_representations]}')
        reduced_representation = self._apply_time_reduce(branch_representations, masks)
        logging.debug(f'representation sizes before concatenation: {[tuple(x.shape) for x in reduced_representation]}')
        combined_features = torch.cat(reduced_representation, dim=-1) # [(B, T, F), ...] -> (B, T, combined_dim) or [(B, F), ...] -> (B, combined_dim)
        logging.debug(f'representation sizes after concatenation: {tuple(combined_features.shape)}')
        fused_representation = self._apply_fusion(combined_features, masks)
        logging.debug(f'fused representation size: {tuple(fused_representation.shape)}')
        outputs = self._apply_output_heads(fused_representation, masks)
        logging.debug(f'output sizes: {[tuple(x.shape) for x in outputs]}')
        return outputs


    def _create_expanded_mask(self, masks: list[torch.BoolTensor], feature_dim: int) -> torch.BoolTensor:
        """
        Create an expanded mask for concatenated tensors based on individual modality masks.

        Args:
            masks (list[torch.BoolTensor]): List of masks, each of shape (B, T).
            feature_sizes (list[int]): List of feature dimensions for each modality.

        Returns:
            torch.BoolTensor: Expanded mask of shape (B, T, F*N).
        """
        expanded_masks = []
        for mask in masks:
            # Expand the mask along the feature dimension
            expanded_mask = mask.unsqueeze(-1).expand(-1, -1, feature_dim // len(masks))  # Shape: (B, T, F)
            expanded_masks.append(expanded_mask)

        # Concatenate expanded masks along the feature axis
        combined_mask = torch.cat(expanded_masks, dim=-1)  # Shape: (B, T, F*N)
        return combined_mask


    def _validate_config(self):
        """Validate the configuration."""
        if self.module_time_reduce not in {None, 'attentionpool', 'gmp', 'gap', 'last'}:
            raise Exception(f'Invalid module_time_reduce: {self.module_time_reduce}')


    def _init_projections(self):
        """Initialize the modality-specific temporal projection layers."""
        self.projectors = nn.ModuleList([
            nn.Conv1d(input_channels, self.d_model, kernel_size=1, padding=0, bias=False)
            for input_channels in self.input_modality_channels
        ])


    def _init_multimodal_signal_module(self, config: dict):
        if self.module_multimodal_signal:
            self.multimodal_signal_aligner = TemporalAlignerFactory.align_time_dim(config)

            self.multimodal_signal_transformer = TransformerEncoder(
                config=config.copy() | {
                    "d_model": (self.n_sequences-1) * self.d_model,
                    "n_layers": config.get("n_layers_mms", 3),
                }
            )
            self.multimodal_signal_projector = nn.Conv1d((self.n_sequences-1) * self.d_model, self.d_model, kernel_size=1, padding=0, bias=False)

    def _init_crossmodal_transformers(self, config: dict):
        """Initialize modality-branches: cross-modal and self-attention transformers."""
        self.modality_indices = range(self.n_sequences)
        self.branch_crossmodal_transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoder(config=config)
                for _ in range(self.n_sequences - 1)
            ])
            for _ in range(self.n_sequences)
        ])
        self.branch_self_attention_transformers = nn.ModuleList([
            TransformerEncoder(
                config=config.copy() | {
                    "d_model": (self.n_sequences - 1) * self.d_model,
                    "n_layers": config.get("n_layers_sa", 3),
                }
            )
            for _ in range(self.n_sequences)
        ])
        
        
    def _init_time_reduce_module(self, config: dict):
        """Initialize the time reduction module."""
        if self.module_time_reduce:
            self.time_reduce_module = TimeReduceFactory.create_time_reduce_layer(config)


    def _init_fusion_module(self, config: dict):
        """Initialize fusion mechanisms."""
        combined_dim = (self.n_sequences - 1) * self.n_sequences * self.d_model

        self.module_self_attention_fusion = config.get("module_self_attention_fusion", False)
        if self.module_self_attention_fusion:
            self.self_attention_fusion_transformer = TransformerEncoder(
                config=config.copy() | {
                    "d_model": combined_dim,
                    "n_layers": config.get("n_layers_sa_fusion", 3),
                }
            )

        self.module_ffn_fusion = config.get("module_ffn_fusion", False)
        if self.module_ffn_fusion:
            self.projection_1 = nn.Linear(combined_dim, combined_dim)
            self.projection_2 = nn.Linear(combined_dim, combined_dim)


    def _init_output_heads(self):
        """Initialize output layers."""
        combined_dim = self.n_sequences * self.d_model if self.n_sequences == 2 else self.n_sequences * (self.n_sequences - 1) * self.d_model
        
        self.output_heads = nn.ModuleList([
            nn.Linear(combined_dim, output_dim)
            for output_dim in self.output_dim
        ]) # list of (B, T, output_dim) or (B, output_dim)


    def _apply_projections(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply temporal convolution projection."""
        return [
            projector(
                F.dropout(
                    x.transpose(1, 2),
                    p=self.dropout_embedding,
                    training=self.training)
                ).transpose(1, 2)
            for projector, x in zip(self.projectors, x_list)
        ]


    def _apply_multimodal_signal(self, x_list: list[torch.Tensor], mask_list: list[torch.BoolTensor]) -> list[torch.Tensor]:
        if self.module_multimodal_signal:
            if mask_list is None:
                mask_list = [torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device) for x in x_list]

            x_new_list, mask_new_list = zip(*[
                self.multimodal_signal_aligner(x, mask)
                for x, mask in zip(x_list, mask_list)
            ])
            mm_x = torch.cat(x_new_list, dim=2)
            mm_mask = torch.stack(mask_new_list, dim=0).any(dim=0)
            mm_x = self.multimodal_signal_transformer(mm_x, query_mask=mm_mask)
            mm_x = self.multimodal_signal_projector(mm_x.transpose(1, 2)).transpose(1, 2)

            x_list.append(mm_x)
            mask_list.append(mm_mask)
        return x_list, mask_list


    def _apply_modality_branch(self,
                               x_list: list[torch.Tensor],
                               mask_list: list[torch.BoolTensor] | None) -> list[torch.Tensor]:
        """Perform modality-branch: cross-modal and self-attention transformers."""
        branch_representations = []
        for target_index in range(self.n_sequences):
            input_indices = [i for i in range(self.n_sequences) if i != target_index]
            cross_modal_hidden = torch.cat([
                self.branch_crossmodal_transformers[target_index][i](
                    x_list[target_index],
                    x_list[input_index],
                    x_list[input_index],
                    query_mask=mask_list[target_index] if mask_list else None,
                    key_mask=mask_list[input_index] if mask_list else None,
                )
                for i, input_index in enumerate(input_indices)
            ], dim=2)

            branch_representations.append(
                self.branch_self_attention_transformers[target_index](cross_modal_hidden, query_mask=mask_list[target_index] if mask_list else None)
            )
        return branch_representations


    def _apply_time_reduce(self, x_list, mask_list):
        """Apply optional time reduction."""
        if self.module_time_reduce: # [(B, T, F), ...] -> [(B, F), ...]
            if mask_list is None: mask_list = [None] * len(x_list)
            x_list = [self.time_reduce_module(x, mask) for x, mask in zip(x_list, mask_list)]
        return x_list

    def _apply_fusion(self, x: torch.Tensor, masks: list[torch.BoolTensor]) -> torch.Tensor:
        """Apply optional fusion mechanisms."""
        if self.module_self_attention_fusion:
            combined_mask = torch.stack(masks, dim=0).all(dim=0) # Combine masks with element-wise AND. [(B, F), ...] -> (B, T)
            x = self.self_attention_fusion_transformer(x, query_mask=combined_mask) # (B, T, F*N) and (B, T)

        if self.module_ffn_fusion:
            x = self.projection_2(
                F.dropout(
                    F.relu(self.projection_1(x)),
                    p=self.dropout_output,
                    training=self.training
                )
            ) + x # ffn + residual

            if masks is not None:
                expanded_mask = self._create_expanded_mask(masks, x.size(-1)) # (B, T, F*N)
                x = x * expanded_mask
        return x


    def _apply_output_heads(self, x: torch.Tensor, masks: torch.BoolTensor) -> list[torch.Tensor]:
        """Apply output heads"""
        if x.ndim == 3 and masks is not None: # (B, F)
            # apply the mask to filter out invalid timesteps in the output
            combined_mask = torch.stack(masks, dim=0).all(dim=0)
            expanded_masks = [combined_mask.unsqueeze(-1).expand(-1, -1, output_dim) for output_dim in self.output_dim]
            return [output_head(x) * mask for output_head, mask in zip(self.output_heads, expanded_masks)]
        
        return [output_head(x) for output_head in self.output_heads]


    @classmethod
    def apply_logit_aggregation(cls, x: list[torch.Tensor], method: str = 'meanpooling') -> list[torch.Tensor]:
        """
        Aggregate logits across the time dimension, ignoring timesteps with all zero features.

        Args:
            x (list[torch.Tensor]): List of tensors, each of shape (B, T, F).
            method (str): Aggregation method. Options are 'meanpooling' or 'maxpooling'.

        Returns:
            list[torch.Tensor]: Aggregated logits, each of shape (B, F).
        """
        if method == 'maxpooling':
            return [
                torch.max(
                    logits.masked_fill(logits.abs().sum(dim=-1, keepdim=True) == 0, float('-inf')), 
                    dim=1
                )[0]
                for logits in x
            ]

        elif method == 'meanpooling':
            return [
                (logits.sum(dim=1) / (logits.abs().sum(dim=-1) > 0).sum(dim=1, keepdim=True).clamp(min=1))
                for logits in x
            ]

        else:
            raise ValueError(f"Method {method} for logit aggregation is not supported.")


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")