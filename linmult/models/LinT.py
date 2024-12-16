import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder
from linmult.models.modules import TRM
from linmult.models.utils import load_config


class LinT(nn.Module):

    def __init__(self, config: dict | str):
        super().__init__()

        if isinstance(config, str):
            config = load_config(config)

        self.input_dim = config.get("input_feature_dim")
        self.output_dim = config.get("output_dim")
        self.d_model = config.get("d_model", 40)
        self.dropout_embedding = config.get("dropout_embedding", 0.1)
        self.dropout_output = config.get("dropout_output", 0.)
        self.module_time_dim_reducer = config.get("time_dim_reducer", None)
        self.module_ffn_fusion = config.get("module_ffn_fusion", None)
        self.special_handling = config.get("special_handling", {})

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
        if self.module_time_dim_reducer:
            self.trm = TRM(config)

        if self.module_ffn_fusion:
            self.projection_1 = nn.Linear(self.d_model, self.d_model)
            self.projection_2 = nn.Linear(self.d_model, self.d_model)

        # 3. Output layer
        self.output_heads = nn.ModuleList([
            nn.Linear(self.d_model, output_dim)
            for output_dim in self.output_dim
        ])


    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None, name: str = None) -> torch.Tensor:
        """input tensor of shape (B, T, F)"""

        if isinstance(x, list):
            if len(x) == 1:
                x = x[0]
            else:
                raise Exception(f'A single tensor is expected got instead {len(x)}.')
            
        if isinstance(mask, list):
            if len(mask) == 1:
                mask = mask[0]
            else:
                raise Exception(f'A single mask is expected got instead {len(mask)}.')

        x = self._apply_projection(x, name=name)

        x = self.self_attention_transformer(x, query_mask=mask) # (B, T, d_model) -> (B, T, d_model)

        if self.module_time_dim_reducer:
            x = self.trm(x, mask) # (B, T, d_model) -> (B, d_model)
            mask = None
        
        if self.module_ffn_fusion:
            x = self.projection_2(
                F.dropout(
                    F.relu(self.projection_1(x)),
                    p=self.dropout_output,
                    training=self.training
                )
            ) + x # ffn + residual

        if mask is not None:
            x = x * mask.unsqueeze(-1) # Mask out padding tokens after residual connection

        outputs = self._apply_output_heads(x, mask) # (B, output_dim) or (B, T, output_dim)
        return outputs


    def _apply_projection(self, x: torch.Tensor, name: str = None) -> torch.Tensor:
        """Apply temporal convolution projection and special handling for specified sequence."""
        if name in self.special_handling:
            params = self.special_handling[name]
            if params['type'] == "weighted_sum" and x.ndim == 4:
                # Handle (B, N, T, F) tensors with weighted aggregation
                x = x[:,params['start_layer']:,:,:]
                weights = F.softmax(self.special_modules[name], dim=0)  # Normalize weights
                weights = weights.to(x.device)
                x = torch.einsum("n,bntf->btf", weights, x)  # Weighted sum: (B, N, T, F) -> (B, T, F)

        x = self.projector(
            F.dropout(
                x.transpose(1, 2),  # (B, T, F) -> (B, F, T)
                p=self.dropout_embedding,
                training=self.training
            )
        ).transpose(1, 2)  # (B, F, T) -> (B, T, d_model)
        return x


    def _apply_output_heads(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> list[torch.Tensor]:
        """Apply output heads"""
        if x.ndim == 3 and mask is not None: # (B, F)
            # apply the mask to filter out invalid timesteps in the output
            expanded_masks = [mask.unsqueeze(-1).expand(-1, -1, output_dim) for output_dim in self.output_dim]
            return [output_head(x) * mask for output_head, mask in zip(self.output_heads, expanded_masks)]
        
        return [output_head(x) for output_head in self.output_heads]