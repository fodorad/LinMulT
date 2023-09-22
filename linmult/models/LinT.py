import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder


class LinT(nn.Module):

    def __init__(self,
                 input_modality_channels: int,
                 output_dim: int,
                 projected_modality_dim: int | list = 40, # d
                 number_of_heads: int = 8,
                 number_of_layers: int = 4, # D
                 embedding_dropout: float = 0.1,
                 cross_attention_dropout: float = 0.1,
                 self_attention_dropout: float = 0.0,
                 relu_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 output_dropout: float = 0.1,
                 attention_mask: bool = True):
        super().__init__()
        self.input_modality_channels = input_modality_channels
        self.output_dim = output_dim
        self.projected_modality_dim = projected_modality_dim
        self.number_of_heads = number_of_heads
        self.number_of_layers = number_of_layers
        self.embedding_dropout = embedding_dropout
        self.cross_attention_dropout = cross_attention_dropout
        self.self_attention_dropout = self_attention_dropout
        self.relu_dropout = relu_dropout
        self.residual_dropout = residual_dropout
        self.output_dropout = output_dropout
        self.attention_mask = attention_mask

        # 1. Temporal convolutional layers
        self.projector = nn.Conv1d(input_modality_channels,
                                   projected_modality_dim,
                                   kernel_size=1,
                                   padding=0,
                                   bias=False)

        # 2. Self Attention Linear Transformer
        self.self_attention_transformer = TransformerEncoder(
            embedding_dim=self.projected_modality_dim,
            number_of_heads=self.number_of_heads,
            number_of_layers=self.number_of_layers,
            attention_dropout=self.self_attention_dropout,
            relu_dropout=self.relu_dropout,
            residual_dropout=self.residual_dropout,
            embedding_dropout=self.self_attention_dropout,
            attention_mask=self.attention_mask)

        # 3. Projection layer
        self.out_layer = nn.Linear(self.projected_modality_dim, self.output_dim)

    def forward(self, input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """input tensor of shape (B, T, F)"""

        if isinstance(input, list):
            if len(input) == 1:
                input = input[0]
            else:
                raise Exception(f'A single tensor is expected got instead {len(input)}.')

        input = input.transpose(1, 2)

        if self.embedding_dropout > 0:
            input = F.dropout(input, p=self.embedding_dropout, training=self.training)

        proj_x = self.projector(input)
        proj_x = proj_x.permute(0, 2, 1)
        hidden_representation = self.self_attention_transformer(proj_x)
        output_seq = self.out_layer(hidden_representation)
        return output_seq