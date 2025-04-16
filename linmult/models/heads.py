import torch.nn as nn
from linmult.models.modules import BN, IN


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

        if config.get("pooling", "gap") == "gap":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif config["pooling"] == "gmp":
            self.pool = nn.AdaptiveMaxPool1d(1)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.get("hidden_dim", 256)),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config.get("hidden_dim", 256), output_dim)
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x) # (B,T,F)

        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.pool(x.permute(0,2,1)).squeeze(-1) # (B,F)
        x = self.proj(x)
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
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(config.get("hidden_dim", 256), output_dim)
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.proj(x)
        return x


class SimpleHead(BaseHead):
    """(B,F) -> (B,output_dim) or (B,T,F) -> (B,T,output_dim)"""

    def __init__(self, input_dim: int, output_dim: int, config: dict):
        super().__init__(input_dim, output_dim, config)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, **kwargs):
        return self.linear(x)


class HeadFactory:

    _registry = {
        "sequence_aggregation": SequenceAggregationHead,
        "sequence": SequenceHead,
        "vector": VectorHead,
        "simple": SimpleHead
    }

    @classmethod
    def register_head(cls, name: str, head_cls: BaseHead):
        cls._registry[name] = head_cls

    @classmethod
    def create_head(cls, name: str, input_dim: int, output_dim: int, config: dict) -> BaseHead:
        if name not in cls._registry:
            raise ValueError(f"Unknown head type: {name}. Registered: {list(cls._registry.keys())}")

        return cls._registry[name].from_config(
            input_dim=input_dim,
            output_dim=output_dim,
            config=config
        )