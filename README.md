<div align="center">

<img src="docs/assets/logo.svg" alt="LinMulT" width="260"/>

<br/>

**General-purpose Multimodal Transformer with Linear-Complexity Attention**

[![CI](https://github.com/fodorad/linmult/workflows/CI/badge.svg)](https://github.com/fodorad/linmult/actions)
[![Coverage](https://codecov.io/gh/fodorad/linmult/branch/main/graph/badge.svg)](https://codecov.io/gh/fodorad/linmult)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=githubpages)](https://adamfodor.com/LinMulT/)
[![PyPI](https://img.shields.io/pypi/v/linmult?color=orange)](https://pypi.org/project/linmult/)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

</div>

---

LinMulT is a modular Transformer library designed for **multimodal sequence modelling**. It handles variable-length inputs across any number of modalities, supports missing-modality scenarios, and offers six attention variants ranging from O(N²) softmax to O(N·s) gated linear attention; all behind a single config file.

## Features

| | |
|---|---|
| **Multiple modalities** | 1–N input sequences with independent lengths and feature dims |
| **Standard attention** | `softmax` — quadratic complexity for baselines and ablations |
| **Efficient attention** | `linear`, `performer`, `flash`, `bigbird` — sub-quadratic complexity |
| **Flexible heads** | sequence, aggregated, upsample, downsample — mix freely |
| **Missing modalities** | zero-mask a modality; model handles it gracefully |
| **Config-driven** | dict or YAML; no subclassing required |

---

## Installation

```bash
pip install linmult
```

For development:

```bash
git clone https://github.com/fodorad/linmult
cd linmult
pip install -e ".[dev,docs]"
make check
```

---

## Quick start

### LinT — single-modality transformer

```python
import torch
from linmult import LinT

x = torch.rand(8, 1500, 25)  # (batch, time, features)

model = LinT({
    'input_feature_dim': 25,
    'heads': [{'name': 'out', 'type': 'simple', 'output_dim': 5}],
    'time_dim_reducer': 'attentionpool',  # aggregate over time
})
result = model(x)
assert result['out'].shape == (8, 5)
```

### LinMulT — multimodal transformer

```python
import torch
from linmult import LinMulT

x1 = torch.rand(8, 1500, 25)  # (batch, time, features)
x2 = torch.rand(8,  450, 35)
x3 = torch.rand(8,  450, 256)

model = LinMulT({
    'input_feature_dim': [25, 35, 256],
    'heads': [{'name': 'sentiment', 'type': 'simple', 'output_dim': 3}],
    'time_dim_reducer': 'gap',
})
result = model([x1, x2, x3])
assert result['sentiment'].shape == (8, 3)
```

### Switching attention type

```python
model = LinT({
    'input_feature_dim': 64,
    'heads': [{'name': 'out', 'type': 'simple', 'output_dim': 10}],
    'attention_type': 'flash',        # linear, performer, flash, bigbird, softmax, mha
    'flash_query_key_dim': 32,        # flash (GAU) scoring dimension
})
```

## Documentation

> [API reference](https://fodorad.github.io/linmult/)

> [Config reference](docs/config-reference.md)

> [Quick-start notebook](examples/quick_start.ipynb)

> [Attention benchmark](examples/benchmark_time_memory.ipynb)

> [UR-Funny training example](examples/benchmark_urfunny.ipynb)


---

## Similar projects using LinMulT

### BlinkLinMulT (2023)

LinMulT trained for blink presence detection and eye state recognition across 7 public benchmark databases.

- Paper: [BlinkLinMulT: Transformer-based Eye Blink Detection](https://www.mdpi.com/2313-433X/9/10/196)
- Code: [github.com/fodorad/BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT)

### PersonalityLinMulT (2022)

LinMulT trained for Big Five personality trait estimation and sentiment analysis (MOSI, MOSEI, First Impressions V2).

- Paper: [Multimodal Sentiment and Personality Perception Under Speech](https://proceedings.mlr.press/v173/fodor22a.html)
- Code: [github.com/fodorad/PersonalityLinMulT](https://github.com/fodorad/PersonalityLinMulT)

---

## Citation

If you found this work helpful, please cite the relevant paper:

**Eye blink detection (2023)**

```bibtex
@article{blinklinmult-fodor23,
  title   = {BlinkLinMulT: Transformer-based Eye Blink Detection},
  author  = {Fodor, {\'A}d{\'a}m and Fenech, Kristian and L{\H{o}}rincz, Andr{\'a}s},
  journal = {Journal of Imaging},
  pages   = {1--19},
  year    = {2023}
}
```

**Personality and sentiment estimation (2022)**

```bibtex
@InProceedings{pmlr-v173-fodor22a,
  title     = {Multimodal Sentiment and Personality Perception Under Speech:
               A Comparison of Transformer-based Architectures},
  author    = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and
               Jacques Junior, Julio C. S. and Escalera, Sergio and
               Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages     = {218--241},
  year      = {2022},
  volume    = {173},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

---

## Contact

**Ádám Fodor** — [adamfodor.com](https://adamfodor.com) · fodorad201@gmail.com
