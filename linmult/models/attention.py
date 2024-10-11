import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFactory:
    """Factory class to create attention layers based on configuration."""
    
    @staticmethod
    def create_attention_layer(d_model: int = 768,
                               n_heads: int = 8,
                               config: dict | None = None):
        attention_type = config.get("attention_type", "bigbird")

        if attention_type not in {"bigbird", "linear", "softmax", "mha"}:
            raise ValueError(f"Given attention_type ({attention_type}) is not supported." \
                             "Choose from {'bigbird', 'linear', 'softmax', 'mha'}.")

        if attention_type == "bigbird":
            return AttentionLayer(
                BigBirdAttention(
                    d_model=d_model,
                    num_heads=n_heads,
                    block_size=config.get("block_size", 64),
                    num_global_tokens=config.get("num_global_tokens", 16),
                    num_random_tokens=config.get("num_random_tokens", 10)
                ),
                d_model=d_model,
                n_heads=n_heads
            )
        elif attention_type == "linear":
            return AttentionLayer(
                LinearAttention(
                    d_model=d_model,
                    num_heads=n_heads,
                    feature_map=config.get("feature_map", None),
                    eps=config.get("eps", 1e-6)
                ),
                d_model=d_model,
                n_heads=n_heads
            )
        elif attention_type == "softmax":
            return AttentionLayer(
                SoftmaxAttention(
                    d_model=d_model,
                    num_heads=n_heads,
                    eps=config.get("eps", 1e-6)
                ),
                d_model=d_model,
                n_heads=n_heads
            )
        else: # mha
            return nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=config.get("dropout_attention", 0.),
                batch_first=True
            )


class AttentionLayer(torch.nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.

    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.

    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
    """
    def __init__(self,
                 attention,
                 d_model: int,
                 n_heads: int,
                 d_keys: int | None = None,
                 d_values: int | None = None):
        super().__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        self.inner_attention = attention
        self.query_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = torch.nn.Linear(d_model, d_values * n_heads)
        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries, keys, values, query_mask=None, key_mask=None):
        """Apply attention to the passed in queries/keys/values after projecting them to multiple heads.

        In the argument description we make use of the following sizes
            - B: the batch size
            - T_1: The maximum length of the queries
            - T_2: The maximum length of the keys (the actual length per sequence is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as 'd_model'

        Args:
            queries: (B, T_1, D) tensor containing the queries
            keys: (B, T_2, D) tensor containing the keys
            values: (B, T_2, D) tensor containing the values
            attn_mask: (B*H, T_1, T_2) combined attention mask
            query_mask: (B, T_1) boolean mask for the queries
            key_mask: (B, T_2) boolean mask for the queries

        Returns:
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        B, T_1, _ = queries.shape
        _, T_2, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(B, T_1, H, -1)
        keys = self.key_projection(keys).view(B, T_2, H, -1)
        values = self.value_projection(values).view(B, T_2, H, -1)

        if query_mask is None and key_mask is None:
            attn_mask = None
        else:
            # Create combined attention mask
            if query_mask is None:
                query_mask = torch.zeros(size=queries.shape[:2]).bool()
            
            if key_mask is None:
                key_mask = torch.zeros(size=keys.shape[:2]).bool()

            attn_mask = query_mask.unsqueeze(-1) * key_mask.unsqueeze(1)
            attn_mask = attn_mask.float() * -1e9 # Convert to float and apply large negative value
            attn_mask = attn_mask.unsqueeze(1) # Shape: (B, 1, T_1, T_2)

        # Compute the attention
        new_values, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        new_values = new_values.view(B, T_1, -1)

        # Project the output and return
        return self.out_projection(new_values), attn


class BigBirdAttention(nn.Module):

    def __init__(self, d_model, num_heads, block_size, num_global_tokens, num_random_tokens, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens
        self.dropout = nn.Dropout(dropout)
        assert self.head_dim * num_heads == d_model, "embed_dim must be divisible by num_heads"

    def forward(self, q, k, v, attn_mask):
        batch_size, tgt_len, n_heads, head_dim = q.size()
        _, src_len, _, _ = k.size()

        attn_output = torch.zeros_like(q) # Shape: (B, tgt_len, num_heads, head_dim)

        # Local attention
        for i in range(0, tgt_len, self.block_size):
            q_block = q[:, i:i+self.block_size, :, :] # Shape: (B, block_size, num_heads, head_dim)
            k_block = k[:, i:i+self.block_size, :, :] # Shape: (B, block_size, num_heads, head_dim)
            v_block = v[:, i:i+self.block_size, :, :] # Shape: (B, block_size, num_heads, head_dim)

            scores = torch.einsum("bnhd,bmhd->bhnm", q_block, k_block) / (self.head_dim ** 0.5) # Shape: (B, num_heads, block_size, block_size)

            if attn_mask is not None:
                attn_mask_block = attn_mask[:, :, i:i+self.block_size, i:i+self.block_size] # Shape: (B, 1, block_size, block_size)
                scores += attn_mask_block

            attn_weights = torch.softmax(scores, dim=-1) # Shape: (B, num_heads, block_size, block_size)
            attn_output[:, i:i+self.block_size, :, :] += torch.einsum("bhnm,bmhd->bnhd", attn_weights, v_block)

        # Global attention
        if self.num_global_tokens > 0:
            global_q = q[:, :self.num_global_tokens, :, :] # Shape: (B, num_global_tokens, num_heads, head_dim)
            global_k = k[:, :self.num_global_tokens, :, :] # Shape: (B, num_global_tokens, num_heads, head_dim)
            global_v = v[:, :self.num_global_tokens, :, :] # Shape: (B, num_global_tokens, num_heads, head_dim)

            scores = torch.einsum("bnhd,bmhd->bhnm", global_q, k) / (self.head_dim ** 0.5) # Shape: (B, num_heads, num_global_tokens, src_len)

            if attn_mask is not None:
                attn_mask_global = attn_mask[:, :, :self.num_global_tokens, :] # Shape: (B, 1, num_global_tokens, src_len)
                scores += attn_mask_global

            global_attn_weights = torch.softmax(scores, dim=-1) # Shape: (B, num_heads, num_global_tokens, src_len)
            attn_output[:, :self.num_global_tokens, :, :] += torch.einsum("bhnm,bnhd->bnhd", global_attn_weights, global_v)

        # Random attention
        rand_indices = torch.randint(0, src_len, (self.num_random_tokens,), device=q.device)
        for idx in rand_indices:
            rand_q = q[:, idx:idx+1, :, :] # Shape: (B, 1, num_heads, head_dim)
            rand_k = k[:, idx:idx+1, :, :] # Shape: (B, 1, num_heads, head_dim)
            rand_v = v[:, idx:idx+1, :, :] # Shape: (B, 1, num_heads, head_dim)

            scores = torch.einsum("bnhd,bmhd->bhnm", rand_q, k) / (self.head_dim ** 0.5) # Shape: (B, num_heads, 1, src_len)

            if attn_mask is not None:
                attn_mask_rand = attn_mask[:, :, idx:idx+1, :] # Shape: (B, 1, 1, src_len)
                scores += attn_mask_rand

            rand_attn_weights = torch.softmax(scores, dim=-1) # Shape: (B, num_heads, 1, src_len)
            attn_output[:, idx:idx+1, :, :] += torch.einsum("bhnm,bmhd->bnhd", rand_attn_weights, rand_v)

        return attn_output.contiguous(), None # Shape: (B, tgt_len, num_heads, head_dim)


class SoftmaxAttention(torch.nn.Module):
    """Implement standard softmax attention with O(N^2 D) complexity.

    Given the queries, keys, and values as Q, K, V, the attention is computed as:

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    where the dot product of Q and K is used to determine the attention scores,
    which are then applied to the values V.

    Arguments
    ---------
        eps: float, a small number to ensure numerical stability in softmax
             (default: 1e-6)
    """
    def __init__(self, d_model: int, num_heads: int, eps: float = 1e-6):
        super(SoftmaxAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.query_dimensions = d_model // num_heads
        self.eps = eps

    def forward(self, queries, keys, values, *args, **kwargs):
        # Compute the dot product between queries and keys
        # Q shape: (B, T_1, H, D), K shape: (B, T_2, H, D)
        # Attention matrix A shape: (B, H, T_1, T_2)
        attention_scores = torch.einsum("nlhd,nshd->nhls", queries, keys)
        
        # Apply scaling for stability
        scaling_factor = self.query_dimensions ** 0.5
        attention_scores = attention_scores / scaling_factor
        
        # Apply the softmax to the attention matrix
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the new values by applying the attention weights to the values
        # V shape: (B, T_2, H, D)
        # Output shape: (B, T_1, H, D)
        V = torch.einsum("nhls,nshd->nlhd", attention_weights, values)

        return V.contiguous(), attention_weights


#########################################################################
#                                                                       #
#   Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/   #
#   Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,  #
#   Apoorv Vyas <avyas@idiap.ch>                                        #
#   https://github.com/idiap/fast-transformers                          #
#                                                                       #
#########################################################################


class LinearAttention(torch.nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, d_model: int, num_heads: int, feature_map=None, eps: float = 1e-6):
        super(LinearAttention, self).__init__()
        query_dimensions = d_model // num_heads
        self.feature_map = feature_map(query_dimensions) if feature_map else elu_feature_map(query_dimensions)
        self.eps = eps

    def forward(self, queries, keys, values, *args, **kwargs):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries) # Shape: (B, T_1, H, D)
        K = self.feature_map.forward_keys(keys) # Shape: (B, T_2, H, D)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous(), None


class FeatureMap(torch.nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)


if __name__ == "__main__":
    from time import time

    d_model = 512
    n_heads = 8
    block_size = 16
    num_global_tokens = 3
    num_random_tokens = 4

    bigbird_attention = AttentionLayer(
        BigBirdAttention(d_model, n_heads, block_size, num_global_tokens, num_random_tokens),
        d_model=d_model,
        n_heads=n_heads
    )

    linear_attention = AttentionLayer(
        LinearAttention(d_model, n_heads),
        d_model=d_model,
        n_heads=n_heads
    )

    mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    queries = torch.randn(16, 300, d_model)
    keys = torch.randn(16, 400, d_model)
    values = torch.randn(16, 400, d_model)
    query_mask = torch.randint(0, 2, (16, 300)).bool()
    key_mask = torch.randint(0, 2, (16, 400)).bool()

    start = time()
    mha_output, _ = mha(queries, keys, values)
    elapsed_linear = time() - start
    print(f"Elapsed time for mha: {elapsed_linear:.4f} seconds")

    start = time()
    linear_output, _ = linear_attention(queries, keys, values)
    elapsed_linear = time() - start
    print(f"Elapsed time for linear_attention: {elapsed_linear:.4f} seconds")
    
    start = time()
    bigbird_output, _ = bigbird_attention(queries, keys, values, query_mask, key_mask)
    elapsed_bigbird = time() - start
    print(f"Elapsed time for bigbird_attention: {elapsed_bigbird:.4f} seconds")
    
    assert bigbird_output.shape == (16,300,512)
    assert linear_output.shape == (16,300,512)
    assert mha_output.shape == (16,300,512)