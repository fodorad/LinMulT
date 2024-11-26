import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder, TimeReduceFactory
from linmult.models.config_loader import load_config


class LinT(nn.Module):

    def __init__(self, config: dict | str):
        super().__init__()

        if isinstance(config, str):
            config = load_config(config)

        self.input_dim = config.get("input_modality_channels")
        self.output_dim = config.get("output_dim")
        self.d_model = config.get("d_model", 40)
        self.dropout_embedding = config.get("dropout_embedding", 0.1)
        self.dropout_relu = config.get("dropout_relu", 0.1)
        self.dropout_residual = config.get("dropout_residual", 0.1)
        self.module_time_reduce = config.get("module_time_reduce", None)

        # 1. Temporal convolutional layers
        self.projector = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.d_model,
            kernel_size=1,
            padding=0,
            bias=False
        ) # (B, C, T) -> (B, d_model, T)

        # 2. Self-Attention Transformer
        self.self_attention_transformer = TransformerEncoder(config)

        # Optional: time reduce module
        if self.module_time_reduce:
            self.time_reduce_module = TimeReduceFactory.create_time_reduce_layer(config)

        # 3. Output layer
        self.output_heads = nn.ModuleList([
            nn.Linear(self.d_model, output_dim)
            for output_dim in self.output_dim
        ])


    def forward(self, input: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """input tensor of shape (B, T, F)"""

        if isinstance(input, list):
            if len(input) == 1:
                input = input[0]
            else:
                raise Exception(f'A single tensor is expected got instead {len(input)}.')
            
        if isinstance(mask, list):
            if len(mask) == 1:
                mask = mask[0]
            else:
                raise Exception(f'A single mask is expected got instead {len(mask)}.')

        input = input.transpose(1, 2) # (B, T, C) -> (B, C, T)

        if self.dropout_embedding > 0:
            input = F.dropout(input, p=self.dropout_embedding, training=self.training)

        proj_x = self.projector(input) # (B, C, T) -> (B, d_model, T)
        proj_x = proj_x.permute(0, 2, 1) # (B, d_model, T) -> (B, T, d_model)
        hidden_representation = self.self_attention_transformer(proj_x, query_mask=mask) # (B, T, d_model) -> (B, T, d_model)

        if self.module_time_reduce:
            hidden_representation = self.time_reduce_module(hidden_representation, mask) # (B, T, d_model) -> (B, d_model)

        outputs = self._apply_output_heads(hidden_representation, mask) # (B, output_dim) or (B, T, output_dim)
        return outputs


    def _apply_output_heads(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> list[torch.Tensor]:
        """Apply output heads"""
        if x.ndim == 3 and mask is not None: # (B, F)
            # apply the mask to filter out invalid timesteps in the output
            expanded_masks = [mask.unsqueeze(-1).expand(-1, -1, output_dim) for output_dim in self.output_dim]
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