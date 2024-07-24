import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from linmult.models.positional_embedding import PositionalEncoding
from linmult.models.attention import AttentionFactory


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers.
    
    Args:
        config_model (dict) with the following key-value pairs:
            d_model (int, optional): input embedding dimension. Defaults to 40.
            n_heads (int, optional): number of heads. Defaults to 6.
            n_layers (int, optional): number of layers. Defaults to 4.
            dropout_embedding (float, optional): dropout applied on the input tensors. Defaults to 0.1.
            dropout_attention (float, optional): dropout applied on the attention weights. Defaults to 0.1.
            dropout_relu (float, optional): dropout applied on the first layer of the residual block. Defaults to 0.1.
            dropout_residual (float, optional): dropout applied on the residual block. Defaults to 0.1.
            attention_type (str): attention type. Currently the following linear-complexity mechanism are supported {"bigbird", "linear"}. Otherwise softmax attention is used.
            block_size (int): block size for BigBird. Defaults to 64.
            num_global_tokens (int): number of global tokens for BigBird. Defaults to 16.
            num_random_tokens (int): number of random tokens for BigBird. Defaults to 10.
            dropout_attention (float, optional): dropout applied on the output of the attention block. Defaults to 0.
    """
    def __init__(self, config: dict | None = None):
        super().__init__()

        if config is None: config = {}

        d_model = config.get("d_model", 40)

        # Embedding scaling and positional encoding
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = PositionalEncoding()

        # Dropout for input embeddings
        self.dropout_embedding = config.get("dropout_embedding", 0.1)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config=config)
            for _ in range(config.get("n_layers", 4))
        ])

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, x_in: torch.Tensor, # (B, T, d_model)
                      x_in_k: torch.Tensor = None,
                      x_in_v: torch.Tensor = None,
                      query_mask: torch.Tensor | None = None,
                      key_mask: torch.Tensor | None = None):
        # Scale and add positional encoding to the input
        x = self.embed_scale * x_in # (B, T, d_model)
        x += self.embed_positions(x) # (B, T, d_model)

        # Apply dropout to the embeddings
        x = F.dropout(x, p=self.dropout_embedding, training=self.training) # (B, T, d_model)

        if x_in_k is not None and x_in_v is not None:
            # Scale and add positional encoding to key and value inputs
            x_k = self.embed_scale * x_in_k + self.embed_positions(x_in_k)
            x_v = self.embed_scale * x_in_v + self.embed_positions(x_in_v)

            # Apply dropout to key and value embeddings
            x_k = F.dropout(x_k, p=self.dropout_embedding, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout_embedding, training=self.training)

            # Pass through the stack of encoder layers
            for layer in self.layers:
                x = layer(x, x_k, x_v, query_mask=query_mask, key_mask=key_mask)
        else:
            # Pass through the stack of encoder layers
            for layer in self.layers:
                x = layer(x, query_mask=query_mask)

        # Final layer normalization
        return self.layer_norm(x)


class TransformerEncoderLayer(nn.Module):
    """Single layer of the Transformer Encoder."""

    def __init__(self, config: dict | None = None):
        super().__init__()

        if config is None: config = {}

        self.n_heads = config.get("n_heads", 8)
        self.d_model = config.get("d_model", 40)
        self.attention_type = config.get("attention_type", "linear")
        self.attention = AttentionFactory.create_attention_layer(self.d_model, self.n_heads, config)

        # Feedforward network layers
        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)

        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(2)])

        # Dropout layers
        self.dropout_relu = config.get("dropout_relu", 0.1)
        self.dropout_residual = config.get("dropout_residual", 0.1)
        self.auto_mask = config.get("auto_mask", False)


    def forward(self, x_q: torch.Tensor,
                      x_k: torch.Tensor | None = None,
                      x_v: torch.Tensor | None = None,
                      query_mask: torch.Tensor | None = None,
                      key_mask: torch.Tensor | None = None):
        """Args:
            x_q (torch.Tensor): input to the layer of shape (B, T_1, F)
            x_k (torch.Tensor): key input to the layer of shape (B, T_2, F)
            x_v (torch.Tensor): value input to the layer of shape (B, T_2, F)
            query_mask (torch.Tensor): attention mask for query of shape (B, T_1)
            key_mask (torch.Tensor): attention mask for key of shape (B, T_2)
            mask_auto (bool): automatically calculate mask. Expected mask value is 0. Defaults to True.

        Returns:
            torch.Tensor: encoded output of shape `(B, T_1, F)`
        """
        residual = x_q
        x_q = self.layer_norms[0](x_q)

        # cross-modal attention
        if x_k is not None and x_v is not None:
            x_k = self.layer_norms[0](x_k) # (B, T_2, F)
            x_v = self.layer_norms[0](x_v) # (B, T_2, F)

            if query_mask is not None and not query_mask.shape == x_q.shape[:2] and isinstance(query_mask, torch.BoolTensor):
                    raise ValueError(f"Expected query mask has shape (B, T_1) and BoolTensor type, got instead: {query_mask.shape} and {type(query_mask)}")
            
            if key_mask is not None and not key_mask.shape == x_k.shape[:2] and isinstance(key_mask, torch.BoolTensor):
                    raise ValueError(f"Expected key mask has (B, T_2) shape and BoolTensor type, got instead: {key_mask.shape} and {type(key_mask)}")
        
            if self.auto_mask:
                query_mask = (x_q == 0).all(dim=2)
                query_mask = query_mask.bool() # (B, T_1)
                key_mask = (x_k == 0).all(dim=2)
                key_mask = key_mask.bool() # (B, T_2)
                query_mask = query_mask.to(x_q.device)
                key_mask = key_mask.to(x_q.device)

            if self.attention_type == "mha":
                x_q, _ = self.attention(x_q, x_k, x_v)
            else:
                x_q, _ = self.attention(x_q, x_k, x_v, query_mask=query_mask, key_mask=key_mask) # returns (B, T_1, F) and (B, T_1, T_2)
        else: # self-attention
            if query_mask is not None and not query_mask.shape == x_q.shape[:2] and isinstance(query_mask, torch.BoolTensor):
                raise ValueError(f"Expected query mask has (B, T) shape and BoolTensor type, got instead: {query_mask.shape} and {type(query_mask)}")

            if self.auto_mask:
                query_mask = (x_q == 0).all(dim=2)
                query_mask = query_mask.bool() # (B, T_1)
                query_mask = query_mask.to(x_q.device)

            if self.attention_type == "mha":
                x_q, _ = self.attention(x_q, x_q, x_q)
            else:
                x_q, _ = self.attention(x_q, x_q, x_q, query_mask=query_mask) # returns (B, T_1, F) and (B, T_1, T_1)

        x_q = F.dropout(x_q, p=self.dropout_residual, training=self.training)
        x_q = residual + x_q

        residual = x_q
        x_q = self.layer_norms[1](x_q)
        x_q = F.relu(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.dropout_relu, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.dropout_residual, training=self.training)
        x_q = residual + x_q
        return x_q


class TimeReduceFactory:
    """Factory class to create modules for reducing the time dimension from configuration."""
    
    @staticmethod
    def create_time_reduce_layer(config: dict | None = None):
        time_reduce_type = config.get("time_reduce_type", "attentionpool1d")

        if time_reduce_type == "attentionpool":
            input_channels = config.get("input_modality_channels")

            if isinstance(input_channels, int):
                d_model = config.get("d_model", 40)
            else:
                d_model = (len(input_channels)-1) * config.get("d_model", 40)
            
            return AttentionPooling(d_model)
        elif time_reduce_type == "gmp":
            return GlobalMaxPooling()
        elif time_reduce_type == "gap":
            return GlobalAvgPooling()
        else: # last
            return LastTimestamp()


class LastTimestamp(nn.Module):

    def forward(self, x):
        return x[:,-1,:] # Shape: (B, T, d_model) -> (B, d_model)

class GlobalAvgPooling(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (B, T, d_model)
        x = x.permute(0, 2, 1)  # Change shape to (B, d_model, T)
        x = self.global_avg_pool(x)  # Shape: (B, d_model, 1)
        x = x.squeeze(-1)  # Shape: (B, d_model)
        return x


class GlobalMaxPooling(nn.Module):

    def __init__(self):
        super().__init__()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x shape: (B, T, d_model)
        x = x.permute(0, 2, 1)  # Change shape to (B, d_model, T)
        x = self.global_max_pool(x)  # Shape: (B, d_model, 1)
        x = x.squeeze(-1)  # Shape: (B, d_model)
        return x


class AttentionPooling(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (B, T, d_model)
        attn_weights = self.attention(x).squeeze(-1)  # Shape: (B, T)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # Shape: (B, T)
        weighted_avg = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # Shape: (B, d_model)
        return weighted_avg


if __name__ == "__main__":
    
    batch_size = 2
    T_1 = 90
    T_2 = 300
    d_model = 40

    x_1 = torch.randn(batch_size, T_1, d_model) # (B, T_1, d_model)
    m_1 = (torch.rand(batch_size, T_1) > 0.5).bool() # (B, T_1)
    x_2 = torch.randn(batch_size, T_2, d_model) # (B, T_2, d_model)
    m_2 = (torch.rand(batch_size, T_2) > 0.5).bool() # (B, T_2)

    layer = TransformerEncoderLayer()
    output_sa = layer(x_1, query_mask=m_1) # self-attention
    output_ca = layer(x_1, x_2, x_2, query_mask=m_1, key_mask=m_2) # cross-attention

    print("input x_1 shape:", x_1.shape)
    print("output sa shape:", output_sa.shape)
    print("input x_2 shape:", x_2.shape)
    print("output ca shape:", output_ca.shape)