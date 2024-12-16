from linmult.models.LinT import LinT
from linmult.models.LinMulT import LinMulT
from linmult.models.utils import load_config, apply_logit_aggregation
from importlib.metadata import version

try:
    __version__ = version("linmult")
except Exception:
    __version__ = "unknown"