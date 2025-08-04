import torch.nn as nn
from linmult.models.modules import BN, IN
from linmult.models.transformer import AttentionPooling


class BaseHead(nn.Module):
    """Base class for all output heads"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__()
        self.config = config

    @classmethod
    def from_config(cls, input_dim: int, output_dim: int, config: dict):
        return cls(input_dim, output_dim, config)


class SequenceAggregationHead(BaseHead):
    """(B,T,F) -> (B,output_dim)"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)
    
        if config.get("norm", "bn") == "bn":
            self.norm = BN(input_dim, time_aware=True)
        elif config["norm"] == "in":
            self.norm = IN(input_dim, time_aware=True)

        self.is_attention_pooling = False
        if config.get("pooling", "gap") == "gap":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif config["pooling"] == "gmp":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif config["pooling"] == "attentionpool":
            self.is_attention_pooling = True
            self.pool = AttentionPooling(config.get("hidden_dim", 256))
        else:
            raise ValueError(f"Unknown pooling type: {config['pooling']}")

        self.proj_1 =  nn.Sequential(
            nn.Linear(input_dim, config.get("hidden_dim", 256)),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(config.get("hidden_dim", 256), output_dim)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x) # (B,T,F)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj_1(x)

        if self.is_attention_pooling:
            x = self.pool(x, mask) # (B,T,F) -> (B,F)
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1) # (B,F)

        x = self.proj_2(x)

        return x


class SequenceHead(BaseHead):
    """(B,T,F) -> (B,T,output_dim)"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)

        if config.get("norm", "bn") == "bn":
            self.norm = BN(input_dim, time_aware=True)
        elif config["norm"] == "in":
            self.norm = IN(input_dim, time_aware=True)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.get("hidden_dim", 256)),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config.get("hidden_dim", 256), output_dim)
        )

    def forward(self, x, mask=None):

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x)
        x = self.proj(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x


class VectorHead(BaseHead):
    """(B,F) -> (B,output_dim)"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)

        if config.get("norm", "bn") == "bn":
            self.norm = BN(input_dim, time_aware=False)
        elif config["norm"] == "in":
            self.norm = IN(input_dim, time_aware=False)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.get("hidden_dim", 256)),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config.get("hidden_dim", 256), output_dim)
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.proj(x)
        return x


class SimpleHead(BaseHead):
    """(B,F) -> (B,output_dim) or (B,T,F) -> (B,T,output_dim) or (B,T,F) -> (B,output_dim)"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)

        pooling = config.get("pool", None)
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "gmp":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pooling == "attentionpool":
            self.pool = AttentionPooling(input_dim)
        else:
            self.pool = None

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, **kwargs):

        if self.pool is not None:
            x = self.pool(x.permute(0,2,1)).squeeze(-1) # (B,F)

        return self.linear(x)


class UpsampleHead(BaseHead):
    """(B,T_in,F) -> (B,time_dim,F_out) with learnable upsampling"""
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)
        self.target_time_dim = config["output_time_dim"]
        input_time_dim = config["input_time_dim"]

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
        )

        # Calculate number of upsampling layers needed
        self.upsample_layers = nn.ModuleList()
        current_dim = input_time_dim
        
        # Keep adding upsampling layers until next upsampling would exceed target
        while current_dim * 2 <= self.target_time_dim:
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose1d(
                    in_channels=output_dim,
                    out_channels=output_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.GELU()
            ))
            current_dim *= 2

        # Final adjustment to target size
        self.final_adjust = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
            nn.AdaptiveAvgPool1d(self.target_time_dim)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj(x)  # (B, T_in, F_out)
        x = x.transpose(1, 2)  # (B, F_out, T_in)

        # Apply upsampling layers
        for layer in self.upsample_layers:
            x = layer(x)

        # Final adjustment to exact size
        x = self.final_adjust(x)
        return x.transpose(1, 2)  # (B, time_dim, F_out)


class DownsampleHead(BaseHead):
    """(B,T_in,F) -> (B,time_dim,F_out) with learnable downsampling"""
    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)
        self.target_time_dim = config["output_time_dim"]
        input_time_dim = config["input_time_dim"]

        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
        )

        # Calculate number of downsampling layers needed
        self.downsample_layers = nn.ModuleList()
        current_dim = input_time_dim
        
        # Keep adding downsampling layers until next downsampling would go below target
        while current_dim // 2 >= self.target_time_dim:
            self.downsample_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=output_dim,
                    out_channels=output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode='replicate'
                ),
                nn.GELU()
            ))
            current_dim = current_dim // 2  # Integer division to track dimension

        # Final adjustment to target size
        self.final_pool = nn.AdaptiveAvgPool1d(self.target_time_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.proj(x)  # (B, T_in, F_out)
        x = x.transpose(1, 2)  # (B, F_out, T_in)

        # Apply downsampling layers
        for layer in self.downsample_layers:
            x = layer(x)

        # Final adjustment to exact size
        x = self.final_pool(x)

        return x.transpose(1, 2)  # (B, time_dim, F_out)


class HeadFactory:

    _registry = {
        "sequence_aggregation": SequenceAggregationHead,
        "sequence": SequenceHead,
        "vector": VectorHead,
        "simple": SimpleHead,
        "upsample": UpsampleHead,
        "downsample": DownsampleHead
    }

    @classmethod
    def register_head(cls, name: str, head_cls: BaseHead):
        cls._registry[name] = head_cls

    @classmethod
    def create_head(cls, type: str, input_dim: int, output_dim: int, config: dict) -> BaseHead:
        if type not in cls._registry:
            raise ValueError(f"Unknown head type: {type}. Registered: {list(cls._registry.keys())}")

        return cls._registry[type].from_config(
            input_dim=input_dim,
            output_dim=output_dim,
            config=config
        )