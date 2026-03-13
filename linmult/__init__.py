"""LinMulT — Linear-complexity Multimodal Transformer.

A modular Transformer library for multimodal sequence modelling.
Handles variable-length inputs across any number of modalities, supports
missing-modality scenarios, and offers six attention variants from O(N²)
softmax to O(N·s) gated linear attention — all behind a single config key.

Public API::

    from linmult import LinMulT, LinT, LinMulTConfig, LinTConfig, HeadConfig, AttentionConfig
"""

from importlib.metadata import version
from pathlib import Path

from linmult.core.attention import AttentionConfig
from linmult.core.config import HeadConfig, LinMulTConfig, LinTConfig
from linmult.core.utils import apply_logit_aggregation, load_config
from linmult.models.LinMulT import LinMulT
from linmult.models.LinT import LinT

__all__ = [
    "LinMulT",
    "LinT",
    "LinMulTConfig",
    "LinTConfig",
    "HeadConfig",
    "AttentionConfig",
    "apply_logit_aggregation",
    "load_config",
]

try:
    __version__ = version("linmult")
except Exception:
    __version__ = "unknown"

PROJECT_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "linmult"
RESOURCE_DIR = PROJECT_ROOT / "resources"
TEST_DIR = PROJECT_ROOT / "tests"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
