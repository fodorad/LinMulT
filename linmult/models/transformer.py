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
            n_heads (int, optional): number of heads. Defaults to 8.
            n_layers (int, optional): number of layers. Defaults to 6.
            dropout_qkv (float, optional): dropout applied on the input tensors. Defaults to 0.
            dropout_relu (float, optional): dropout applied on the first layer of the residual block. Defaults to 0.1.
            dropout_residual (float, optional): dropout applied on the residual block. Defaults to 0.1.
            attention_type (str): attention type. Currently the following linear-complexity mechanism are supported {"bigbird", "linear"}. Otherwise softmax attention is used.
            block_size (int): block size for BigBird. Defaults to 64.
            num_global_tokens (int): number of global tokens for BigBird. Defaults to 16.
            num_random_tokens (int): number of random tokens for BigBird. Defaults to 10.
    """
    def __init__(self, config: dict | None = None):
        super().__init__()

        if config is None: config = {}

        d_model = config.get("d_model", 40)

        # Embedding scaling and positional encoding
        self.embed_scale = math.sqrt(d_model)
        self.embed_positions = PositionalEncoding(dropout=config.get("dropout_pe", 0.))

        # Dropout for input embeddings
        self.dropout_qkv = config.get("dropout_qkv", 0.)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config=config)
            for _ in range(config.get("n_layers", 6))
        ])

        # Final layer normalization
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, 
            x_q: torch.Tensor,
            x_k: torch.Tensor | None = None,
            x_v: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            key_mask: torch.Tensor | None = None
        ) -> torch.Tensor:
        """Transformer Encoder inference.

        Self-Attention transformer encoder: x_q == x_k == x_v
        CrossModal transformer encoder: x_q != x_k, x_q != x_v, x_k == x_v
        
        Note: 
            x_k and x_v is not provided, then x_q == x_k == x_v.

        Args:
            x_q (torch.Tensor): input to the layer of shape (B, T_1, F)
            x_k (torch.Tensor, Optional): key input to the layer of shape (B, T_2, F)
            x_v (torch.Tensor, Optional): value input to the layer of shape (B, T_2, F)
            query_mask (torch.Tensor, Optional): attention mask for query of shape (B, T_1)
            key_mask (torch.Tensor, Optional): attention mask for key of shape (B, T_2)

        Returns:
            (torch.Tensor): encoded output of shape (B, T_1, F)
        """

        # Scale and add positional encoding to the input
        x = self.embed_scale * x_q # (B, T, d_model)
        x += self.embed_positions(x) # (B, T, d_model)

        # Apply dropout to the embeddings
        x = F.dropout(x, p=self.dropout_qkv, training=self.training) # (B, T, d_model)

        if x_k is not None and x_v is not None:
            # Scale and add positional encoding to key and value inputs
            x_k = self.embed_scale * x_k + self.embed_positions(x_k)
            x_v = self.embed_scale * x_v + self.embed_positions(x_v)

            # Apply dropout to key and value embeddings
            x_k = F.dropout(x_k, p=self.dropout_qkv, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout_qkv, training=self.training)

            # Pass through the stack of encoder layers
            for layer in self.layers:
                x = layer(x, x_k, x_v, query_mask=query_mask, key_mask=key_mask)
        else:
            # Pass through the stack of encoder layers
            for layer in self.layers:
                x = layer(x, query_mask=query_mask)

        # Final layer normalization
        x = self.layer_norm(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """Single layer of the Transformer Encoder."""

    def __init__(self, config: dict | None = None):
        super().__init__()

        if config is None: config = {}

        self.name = config.get("name", "unknown")
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


    def forward(self, 
            x_q: torch.Tensor,
            x_k: torch.Tensor | None = None,
            x_v: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            key_mask: torch.Tensor | None = None
        ):
        """Transformer Encoder layer inference.
        
        Args:
            x_q (torch.Tensor): input to the layer of shape (B, T_1, F)
            x_k (torch.Tensor): key input to the layer of shape (B, T_2, F)
            x_v (torch.Tensor): value input to the layer of shape (B, T_2, F)
            query_mask (torch.Tensor): attention mask for query of shape (B, T_1)
            key_mask (torch.Tensor): attention mask for key of shape (B, T_2)

        Returns:
            (torch.Tensor): encoded output of shape (B, T_1, F)
        """
        if query_mask is not None and not query_mask.shape == x_q.shape[:2] and isinstance(query_mask, torch.BoolTensor):
            raise ValueError(f"Expected query mask has shape (B, T_1) and BoolTensor type, got instead: {query_mask.shape} and {type(query_mask)}")

        if key_mask is not None and not key_mask.shape == x_k.shape[:2] and isinstance(key_mask, torch.BoolTensor):
            raise ValueError(f"Expected key mask has (B, T_2) shape and BoolTensor type, got instead: {key_mask.shape} and {type(key_mask)}")

        residual = x_q
        x_q = self.layer_norms[0](x_q)

        # cross-modal attention
        if x_k is not None and x_v is not None:
            x_k = self.layer_norms[0](x_k) # (B, T_2, F)
            x_v = self.layer_norms[0](x_v) # (B, T_2, F)

            if self.attention_type == "mha":
                x_q, _ = self.attention(x_q, x_k, x_v)
            else:
                x_q, _ = self.attention(x_q, x_k, x_v, query_mask=query_mask, key_mask=key_mask) # returns (B, T_1, F) and (B, T_1, T_2)

            if key_mask is not None:
                fully_masked_keys = ~key_mask.any(dim=1)  # Shape: (B,)
                if fully_masked_keys.any():
                    # If all keys are masked for any sample, return zeros for those samples
                    zero_output = torch.zeros_like(x_q, device=x_q.device)
                    x_q = torch.where(fully_masked_keys.unsqueeze(1).unsqueeze(2), zero_output, x_q)

        else: # self-attention

            if self.attention_type == "mha":
                x_q, _ = self.attention(x_q, x_q, x_q)
            else:
                x_q, _ = self.attention(x_q, x_q, x_q, query_mask=query_mask, key_mask=query_mask) # returns (B, T_1, F) and (B, T_1, T_1)

        if query_mask is not None:
            fully_masked_queries = ~query_mask.any(dim=1)  # Shape: (B,)
            if fully_masked_queries.any():
                # Create a tensor of zeros for fully masked samples
                zero_output = torch.zeros_like(x_q, device=x_q.device)
                # Replace the output for fully masked samples with zeros
                x_q = torch.where(fully_masked_queries.unsqueeze(1).unsqueeze(2), zero_output, x_q)

        x_q = F.dropout(x_q, p=self.dropout_residual, training=self.training)
        x_q = residual + x_q

        residual = x_q
        x_q = self.layer_norms[1](x_q)

        x_q = F.gelu(self.fc1(x_q))
        x_q = F.dropout(x_q, p=self.dropout_relu, training=self.training)
        x_q = self.fc2(x_q)
        x_q = F.dropout(x_q, p=self.dropout_residual, training=self.training)
        x_q = residual + x_q

        return x_q


class TemporalFactory:
    """Factory class to create a temporal signal aligner or reducer from configuration."""
    
    @staticmethod
    def time_dim_aligner(method: str = "aap"):
        if method == "aap":
            return AdaptiveAvgPooling()
        elif method == "amp":
            return AdaptiveMaxPooling()
        else: # padding
            return TemporalPadding()

    @staticmethod
    def time_dim_reducer(config: dict | None = None):
        method = config.get("time_dim_reducer", "attentionpool")

        if method == "attentionpool":
            input_feature_dim = config.get("input_feature_dim")

            if isinstance(input_feature_dim, int):
                d_model = config.get("d_model", 40)
            else:
                multiplier = len(input_feature_dim) if config.get("multimodal_signal", False) else (len(input_feature_dim)-1)
                d_model = multiplier * config.get("d_model", 40) 

            return AttentionPooling(d_model)
        elif method == "gmp":
            return GlobalMaxPooling()
        elif method == "gap":
            return GlobalAvgPooling()
        else: # last
            return LastTimestamp()


class TemporalPadding(nn.Module):

    def forward(self, x: torch.Tensor, time_dim: int, mask: torch.BoolTensor = None):
        """Adjusts the time dimension of the input tensor by truncation or padding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            time_dim (int): padding is applied until time_dim size is reached.
            mask (torch.BoolTensor, optional): Mask tensor of shape (B, T). True indicates valid tokens.

        Returns:
            Tuple[torch.Tensor, torch.BoolTensor]: Output tensor and mask, both of shape (B, T_new, F) or (B, T_new).
        """
        current_time_dim = x.size(1)  # Current time dimension (T)

        if current_time_dim > time_dim:
            # Truncate if the current time dimension is larger
            x_new = x[:, :time_dim, :]
            if mask is not None:
                mask_new = mask[:, :time_dim]
            else:
                mask_new = torch.ones((x.size(0), time_dim), dtype=torch.bool, device=x.device)

        elif current_time_dim < time_dim:
            # Pad if the current time dimension is smaller
            pad_size = time_dim - current_time_dim
            x_new = F.pad(x, (0, 0, 0, pad_size))  # Pad along the time dimension (T)

            if mask is not None:
                mask_new = F.pad(mask, (0, pad_size), value=False)  # Extend the mask with False
            else:
                mask_new = torch.ones((x.size(0), current_time_dim), dtype=torch.bool, device=x.device)
                mask_new = F.pad(mask_new, (0, pad_size), value=False)  # Extend the new mask

        else:
            # If already the correct size, do nothing
            x_new = x
            mask_new = mask if mask is not None else torch.ones((x.size(0), current_time_dim), dtype=torch.bool, device=x.device)

        return x_new, mask_new


class AdaptiveMaxPooling(nn.Module):

    def forward(self, x: torch.Tensor, time_dim: int, mask: torch.BoolTensor = None):
        # x: (B, T, F)
        if mask is not None:
            # Expand the mask to match feature dimensions
            expanded_mask = mask.unsqueeze(-1).expand_as(x)  # Shape: (B, T, F)

            # Replace masked positions with a very large negative value (-inf)
            x = x.masked_fill(~expanded_mask, float('-inf'))

        # Apply adaptive max pooling
        x_new = F.adaptive_max_pool1d(x.transpose(1, 2), time_dim).transpose(1, 2)

        if mask is not None:
            mask_new = (x_new != float('-inf')).any(dim=-1)  # Shape: (B, T_new)
            # Replace -inf with zeros in the output
            x_new[x_new == float('-inf')] = 0.0
        else:
            mask_new = torch.ones(x_new.size(0), x_new.size(1), dtype=torch.bool, device=x_new.device)

        return x_new, mask_new


class AdaptiveAvgPooling(nn.Module):

    def forward(self, x: torch.Tensor, time_dim: int, mask: torch.BoolTensor = None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            time_dim (int): padding is applied until time_dim size is reached.
            mask (torch.BoolTensor, optional): Mask tensor of shape (B, T). True indicates valid tokens.

        Returns:
            Tuple[torch.Tensor, torch.BoolTensor]: Output tensor and new mask, both of shape (B, T_new, F).
        """
        if mask is not None:
            # Expand the mask to match feature dimensions
            expanded_mask = mask.unsqueeze(-1).expand_as(x)  # Shape: (B, T, F)

            # Replace masked positions with zero
            x = x.masked_fill(~expanded_mask, 0.0)

            # Generate new mask to track valid segments
            pooled_mask = F.adaptive_avg_pool1d(mask.float().unsqueeze(1), time_dim).squeeze(1)  # Shape: (B, T_new)
            mask_new = pooled_mask > 1e-8  # Binarize the pooled mask

            # Apply adaptive average pooling to the tensor
            x_pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), time_dim).transpose(1, 2)  # Shape: (B, T_new, F)

            # Normalize the pooled tensor to account for masked regions
            # Avoid division by zero by adding a small epsilon
            epsilon = 1e-8
            x_new = x_pooled / (pooled_mask.unsqueeze(-1) + epsilon)  # Shape: (B, T_new, F)
        else:
            # If no mask is provided, perform standard adaptive average pooling
            x_new = F.adaptive_avg_pool1d(x.transpose(1, 2), time_dim).transpose(1, 2)  # Shape: (B, T_new, F)
            mask_new = torch.ones(x.size(0), time_dim, dtype=torch.bool, device=x.device)  # Full valid mask

        return x_new, mask_new


class LastTimestamp(nn.Module):

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Extract the last valid timestamp based on the mask.
        If no mask is provided, select the last timestamp across all time steps.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model).
            mask (torch.BoolTensor, optional): Boolean mask of shape (B, T). Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (B, d_model).
        """
        if mask is not None:
            last_timestamp_index = (mask.sum(dim=1) - 1).long()  # Get the index of the last valid timestamp
        else:
            last_timestamp_index = torch.full((x.size(0),), x.size(1) - 1, dtype=torch.long, device=x.device)  # Use the last index
        
        batch_indices = torch.arange(x.size(0), device=x.device)  # Batch indices
        x = x[batch_indices, last_timestamp_index]  # Gather the last valid timestamp
        return x  # Shape: (B, T, d_model) -> (B, d_model)


class GlobalAvgPooling(nn.Module):

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        # x shape: (B, T, d_model), mask shape: (B, T)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Shape: (B, T, 1)
            x = x * mask  # Zero out padded values
            sum_x = x.sum(dim=1)  # Sum over the time dimension: (B, d_model)
            count_x = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero: (B, 1)
            return sum_x / count_x  # Compute mean: (B, d_model)
        else:
            return x.mean(dim=1)  # Default mean over the time dimension: (B, d_model)


class GlobalMaxPooling(nn.Module):

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        # x shape: (B, T, d_model), mask shape: (B, T)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Shape: (B, T, 1)
            x = x.masked_fill(~mask, float('-inf'))  # Ignore padding with -inf
        return x.max(dim=1)[0]  # Max pooling over the time dimension: (B, d_model)


class AttentionPooling(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        # x shape: (B, T, d_model), mask shape: (B, T)
        attn_weights = self.attention(x).squeeze(-1)  # Shape: (B, T)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))  # Ignore padding with -inf
        attn_weights = torch.softmax(attn_weights, dim=-1)  # Shape: (B, T)
        weighted_avg = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # Shape: (B, d_model)
        return weighted_avg


if __name__ == '__main__':
    x = torch.randn(8, 300, 40)
    mask = torch.ones((8, 300), dtype=bool)
    ap = AttentionPooling(40)
    print(ap(x, mask).shape)