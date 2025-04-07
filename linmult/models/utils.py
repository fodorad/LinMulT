import yaml
import torch


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def apply_logit_aggregation(x: torch.Tensor, mask: torch.BoolTensor | None = None, method: str = 'meanpooling') -> torch.Tensor:
    """Aggregate logits across the time dimension, using only the timesteps indicated by the mask.

    Args:
        x (torch.Tensor): Tensor of shape (B, T, F).
        mask (torch.BoolTensor): Boolean mask of shape (B, T), where True indicates valid timesteps.
        method (str): Aggregation method. Options are 'meanpooling' or 'maxpooling'.

    Returns:
        torch.Tensor: Aggregated logits of shape (B, F).
    """
    if mask is None: mask = torch.ones(size=x.shape[:2], dtype=bool) # (B, T)

    if method == 'maxpooling':
        # Mask invalid timesteps with -inf, then compute max along the time dimension
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # Shape: (B, T, F)
        return torch.max(x_masked, dim=1)[0]  # Shape: (B, F)

    elif method == 'meanpooling':
        # Mask invalid timesteps with 0, then compute weighted sum and normalize by valid counts
        x_masked = x.masked_fill(~mask.unsqueeze(-1), 0.0)  # Shape: (B, T, F)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Shape: (B, 1)
        return x_masked.sum(dim=1) / valid_counts  # Shape: (B, F)

    else:
        raise ValueError(f"Method {method} for logit aggregation is not supported.")