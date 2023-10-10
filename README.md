# LinMulT
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

General-purpose Multimodal Transformer with Linear Complexity Attention Mechanism.

# Setup
### Install package from PyPI
```
pip install linmult
```

### Install package for development
```
git clone https://github.com/fodorad/LinMulT
cd LinMulT
pip install -e .
pip install -U -r requirements.txt
python -m unittest
```

# Quick start
### Example 1:
Simple transformer encoder with linear attention.
The forward pass is performed using an input sequence.
```
import torch
from linmult import LinT

# input shape: (batch_size, time_dimension, feature_dimension)
x = torch.rand((32, 15, 1024), device='cuda')
model = LinT(input_modality_channels=1024, output_dim=5).cuda()
y_pred_seq = model(x)

# output shape: (batch_size, time_dimension, output_dimension)
assert y_pred_seq.size() == torch.Size([32, 15, 5])
```

### Example 2:
Multimodal Transformer with Linear Attention.
The forward pass is performed using 2 input sequences. Both input sequences have the same time dimension.
```
import torch
from linmult import LinMulT

# input shape: (batch_size, time_dimension, feature_dimension)
x_1 = torch.rand((32, 15, 1024), device='cuda')
x_2 = torch.rand((32, 15, 160), device='cuda')
model = LinMulT(input_modality_channels=[1024, 160], output_dim=5).cuda()
y_pred_cls, y_pred_seq = model([x_1, x_2])

# 1. output shape: (batch_size, output_dimension)
assert y_pred_cls.size() == torch.Size([32, 5])

# 2. output shape: (batch_size, time_dimension, output_dimension)
assert y_pred_seq.size() == torch.Size([32, 15, 5])
```

### Example 3:
Multimodal Transformer with Linear Attention. The forward pass is performed using 3 input sequences with different time dimensions.
```
import torch
from linmult import LinMulT

# input shape: (batch_size, time_dimension, feature_dimension)
x_1 = torch.rand((16, 1500, 25), device='cuda')
x_2 = torch.rand((16, 450, 35), device='cuda')
x_3 = torch.rand((16, 120, 768), device='cuda')
model = LinMulT(input_modality_channels=[25, 35, 768],
                output_dim=5,
                add_time_collapse=True,
                add_self_attention_fusion=False).cuda()
y_pred_cls = model([x_1, x_2, x_3])

# output shape: (batch_size, output_dimension)
assert y_pred_cls.size() == torch.Size([16, 5])
```

# Similar projects using LinMulT

### (2023) BlinkLinMulT
LinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.
* paper: BlinkLinMulT: Transformer-based Eye Blink Detection (accepted, available soon)
* code: https://github.com/fodorad/BlinkLinMulT

### (2022) PersonalityLinMulT
LinMulT is trained for Big Five personality trait estimation using the First Impressions V2 dataset and sentiment estimation using the MOSI and MOSEI datasets.
* paper: Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures ([pdf](https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf), [website](https://proceedings.mlr.press/v173/fodor22a.html))
* code: https://github.com/fodorad/PersonalityLinMulT


# Citation - BibTex
If you found our research helpful or influential please consider citing:

### (2023) LinMulT for blink presence detection and eye state recognition:
```
@article{blinklinmult-fodor23,
  title = {BlinkLinMulT: Transformer-based Eye Blink Detection},
  author = {Fodor, {\'A}d{\'a}m and Fenech, Kristian and L{\H{o}}rincz, Andr{\'a}s},
  journal = {...}
  pages = {1--19},
  year = {2023}
}
```

### (2022) LinMulT for personality trait and sentiment estimation:
```
@InProceedings{pmlr-v173-fodor22a,
  title = {Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures},
  author = {Fodor, {\'A}d{\'a}m and Saboundji, Rachid R. and Jacques Junior, Julio C. S. and Escalera, Sergio and Gallardo-Pujol, David and L{\H{o}}rincz, Andr{\'a}s},
  booktitle = {Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages = {218--241},
  year = {2022},
  editor = {Palmero, Cristina and Jacques Junior, Julio C. S. and Clapés, Albert and Guyon, Isabelle and Tu, Wei-Wei and Moeslund, Thomas B. and Escalera, Sergio},
  volume = {173},
  series = {Proceedings of Machine Learning Research},
  month = {16 Oct},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v173/fodor22a/fodor22a.pdf},
  url = {https://proceedings.mlr.press/v173/fodor22a.html}
}
```

# Acknowledgement
The code is inspired by the following two materials:

### Multimodal Transformer:
* paper: Multimodal Transformer for Unaligned Multimodal Language Sequences ([1906.00295](https://arxiv.org/pdf/1906.00295.pdf))
* code: https://github.com/yaohungt/Multimodal-Transformer

### Linear Attention:
* paper: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention ([2006.16236](https://arxiv.org/pdf/2006.16236.pdf))
* code: https://github.com/idiap/fast-transformers

# Contact
* Ádám Fodor (foauaai@inf.elte.hu)