from pathlib import Path
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
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 4)
        self.dropout_embedding = config.get("dropout_embedding", 0.1)
        self.dropout_ca = config.get("dropout_ca", 0.1)
        self.dropout_sa = config.get("dropout_sa", 0.1)
        self.dropout_relu = config.get("dropout_relu", 0.1)
        self.dropout_residual = config.get("dropout_residual", 0.1)


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
        self.add_time_collapse = False
        if config.get("time_reduce_type", None) is not None:
            self.time_reduce_module = TimeReduceFactory.create_time_reduce_layer(config)
            self.add_time_collapse = True

        # 3. Output layer
        self.out_heads = nn.ModuleList([
            nn.Linear(self.d_model, output_dim)
            for output_dim in self.output_dim
        ])

    def forward(self, input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """input tensor of shape (B, T, F)"""

        if isinstance(input, list):
            if len(input) == 1:
                input = input[0]
            else:
                raise Exception(f'A single tensor is expected got instead {len(input)}.')

        input = input.transpose(1, 2) # (B, T, C) -> (B, C, T)

        if self.dropout_embedding > 0:
            input = F.dropout(input, p=self.dropout_embedding, training=self.training)

        proj_x = self.projector(input) # (B, C, T) -> (B, d_model, T)
        proj_x = proj_x.permute(0, 2, 1) # (B, d_model, T) -> (B, T, d_model)
        hidden_representation = self.self_attention_transformer(proj_x) # (B, T, d_model) -> (B, T, d_model)

        if self.add_time_collapse:
            hidden_representation = self.time_reduce_module(hidden_representation) # (B, T, d_model) -> (B, d_model)

        output_cls = [out_layer(hidden_representation) for out_layer in self.out_heads] # (B, output_dim) or (B, T, output_dim)
        return output_cls


if __name__ == "__main__":

    x = torch.randn(16, 90, 256) # (B, T, F)

    model = LinT(
        config={
            'input_modality_channels': x.shape[-1],
            'output_dim': (5,),
        },
    )

    output = model(x)

    print("x shape:", x.shape)
    print("output shape:", output[0].shape)


    model = LinT(
        config={
            'input_modality_channels': x.shape[-1],
            'output_dim': (5,),
            'time_reduce_type': 'attentionpool'
        },
    )

    output = model(x)

    print("x shape:", x.shape)
    print("output shape:", output[0].shape)