"""Utility functions: config loading and logit aggregation."""

from pathlib import Path

import torch
import yaml


def load_config(config_file: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        config_file (str | Path): Path to the YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config


def apply_logit_aggregation(
    x: torch.Tensor, mask: torch.Tensor | None = None, method: str = "meanpooling"
) -> torch.Tensor:
    """Aggregate logits across the time dimension.

    Only timesteps where the mask is ``True`` (valid) contribute to the result.
    Fully-masked samples (all ``False``) return a zero vector.

    Args:
        x (torch.Tensor): Logit tensor of shape ``(B, T, F)``.
        mask (torch.Tensor, optional): Boolean validity mask of shape ``(B, T)``.
            ``True`` = valid timestep. If ``None``, all timesteps are treated as valid.
        method (str): Aggregation method. One of:

            - ``"meanpooling"``: Masked mean over the time dimension.
            - ``"maxpooling"``: Masked max over the time dimension.

    Returns:
        torch.Tensor: Aggregated output of shape ``(B, F)``.

    Raises:
        ValueError: If ``method`` is not one of the supported values.
    """
    m: torch.Tensor = (
        mask
        if mask is not None
        else torch.ones(size=x.shape[:2], dtype=torch.bool, device=x.device)  # (B, T)
    )

    if method == "maxpooling":
        x_masked = x.masked_fill(~m.unsqueeze(-1), float("-inf"))  # (B, T, F)
        result = torch.max(x_masked, dim=1)[0]  # (B, F)
        fully_masked = ~m.any(dim=1)  # (B,)
        if fully_masked.any():
            result = result.masked_fill(fully_masked.unsqueeze(-1), 0.0)
        return result

    elif method == "meanpooling":
        x_masked = x.masked_fill(~m.unsqueeze(-1), 0.0)  # (B, T, F)
        valid_counts = m.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        return x_masked.sum(dim=1) / valid_counts  # (B, F)

    else:
        raise ValueError(f"Method {method} for logit aggregation is not supported.")
