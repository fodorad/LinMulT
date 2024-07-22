from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder
from linmult.models.config_loader import load_config


class LinT(nn.Module):

    def __init__(self, config: dict | str):
        super().__init__()

        if isinstance(config, str) and Path(config).exists():
            config = load_config(config)

        self.input_dim = config.get("input_dim")
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

        # 3. Projection layer
        self.out_layer = nn.Linear(self.d_model, self.output_dim)

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
        output_seq = self.out_layer(hidden_representation) # (B, T, output_dim)
        return output_seq


if __name__ == "__main__":

    x = torch.randn(16, 90, 256) # (B, T, F)

    model = LinT(
        config={
            'input_dim': x.shape[-1],
            'output_dim': 5,
        },
    )

    output = model(x)

    print("x shape:", x.shape)
    print("output shape:", output.shape)