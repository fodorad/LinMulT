#############################################################
#                                                           #
#   Code is inspired by the following repositories:         #
#   1) https://github.com/yaohungt/Multimodal-Transformer   #
#   2) https://github.com/idiap/fast-transformers           #
#                                                           #
#############################################################
import math
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.position_embedding import SinusoidalPositionalEmbedding
from linmult.models.masking import LengthMask, FullMask
from linmult.models.linear_attention import AttentionLayer, LinearAttention


class TransformerEncoder(nn.Module):
    """Transformer encoder.

    Args:
        embedding_dim (int): input embedding dimension
        number_of_heads (int): number of heads
        number_of_layers (int): number of layers
        embedding_dropout (float): dropout applied on the input tensors
        attention_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        residual_dropout (float): dropout applied on the residual block
        attention_mask (bool): whether to apply mask on the attention weights
        attention_type (str): linear or softmax attention mechanism
    """

    def __init__(self, embedding_dim: int,
                 number_of_heads: int,
                 number_of_layers: int,
                 embedding_dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 relu_dropout: float = 0.0,
                 residual_dropout: float = 0.0,
                 attention_mask: bool = False):
        super().__init__()
        self.embed_dim = embedding_dim
        self.embed_scale = math.sqrt(embedding_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embedding_dim)
        self.dropout = embedding_dropout
        self.attn_dropout = attention_dropout
        self.attn_mask = attention_mask
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim=embedding_dim,
                                    number_of_heads=number_of_heads,
                                    attention_dropout=attention_dropout,
                                    relu_dropout=relu_dropout,
                                    residual_dropout=residual_dropout,
                                    attention_mask=attention_mask)
            for _ in range(number_of_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))


    def forward(self, x_in: torch.Tensor, x_in_k: torch.Tensor | None = None, x_in_v: torch.Tensor | None = None):
        """
        Args:
            x_in_q (torch.Tensor): embedded input
            x_in_k (torch.Tensor): embedded input
            x_in_v (torch.Tensor): embedded input

        Returns:
            (torch.Tensor): the last encoder layer's output
        """
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in[:, :, 0])
        x = F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions
            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1) # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1) # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        return self.layer_norm(x)

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`.

    Args:
        embedding_dim (int): input embedding dimension
        number_of_heads (int): number of heads
        attention_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        residual_dropout (float): dropout applied on the residual block
        attention_mask (bool): whether to apply mask on the attention weights
        attention_type (str): linear or softmax attention mechanism
    """
    def __init__(self, embedding_dim: int,
                 number_of_heads: int = 4,
                 attention_dropout: float = 0.1,
                 relu_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 attention_mask: bool = None):
        super().__init__()
        self.embed_dim = embedding_dim
        self.num_heads = number_of_heads

        self.self_attn = AttentionLayer(
            LinearAttention(query_dimensions=embedding_dim//number_of_heads),
            d_model=embedding_dim,
            n_heads=number_of_heads,
            d_keys=embedding_dim//number_of_heads,
            d_values=embedding_dim//number_of_heads
        )

        self.attn_mask = attention_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = residual_dropout

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (torch.Tensor): input to the layer of shape (T, B, F)
            x_k (torch.Tensor): input to the layer of shape (T, B, F)
            x_v (torch.Tensor): input to the layer of shape (T, B, F)
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.layer_norms[0](x)
        N = x.shape[0]
        L = x.shape[1]

        attn_mask = FullMask(L, device=x.device)

        if x_k is None and x_v is None:
            length_mask = LengthMask(x.new_full((N,), L, dtype=torch.int64))
            x = self.self_attn(queries=x, keys=x, values=x, query_lengths=length_mask , key_lengths=length_mask, attn_mask=attn_mask)
        else:
            S = x_k.shape[1]
            length_mask = LengthMask(x.new_full((N,), S, dtype=torch.int64))
            x_k = self.layer_norms[0](x_k)
            x_v = self.layer_norms[0](x_v)
            x = self.self_attn(queries=x, keys=x_k, values=x_v, query_lengths=length_mask , key_lengths=length_mask,  attn_mask=attn_mask)

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.layer_norms[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x


def Linear(in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m