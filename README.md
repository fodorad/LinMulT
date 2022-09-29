# LinearMulT

General-purpose Multimodal Transformer with Linear Attention.

PyTorch implementation of the linear transformer used in the following paper: **Multimodal Sentiment and Personality Perception Under Speech: A Comparison of Transformer-based Architectures** (https://proceedings.mlr.press/v173/fodor22a.html)


# Setup

## Environment
* Python 3.10
* PyTorch and cuDNN 1.12.1+cu102

## Install dependencies
```
conda create -n py310 python=3.10
conda activate py310
pip install -U -r requirements.txt
```

## Install package from repository root
```
pip install -e .
```


# Quick start
The following code builds a multimodal transformer with linear attention, then a forward pass is executed using 3 input sequences.

```
import torch
from linear_mult.models.MulT import MulT

model = MulT(input_modality_channels=[35, 25, 768], 
             output_dim=5, 
             attention_type='linear').cuda()

# input shape: (batch_size, time_dimension, feature_dimension)
x_a = torch.rand((16, 1500, 35), device='cuda')
x_v = torch.rand((16, 450,  25), device='cuda')
x_t = torch.rand((16, 105, 768), device='cuda')

# output shape: (batch_size, output_dimension)
y_true = torch.rand((16, 5), device='cuda')
y_pred = model([x_a, x_v, x_t])

assert y_pred.size() == torch.Size([16, 5])
```

# Run tests
```
python -m unittest
```

# Acknowledgement
The code is heavily built upon the following two repositories:

Multimodal Transformer:
* paper: Multimodal Transformer for Unaligned Multimodal Language Sequences ([1906.00295](https://arxiv.org/pdf/1906.00295.pdf))
* code: https://github.com/yaohungt/Multimodal-Transformer

Linear Attention:
* paper: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention ([2006.16236](https://arxiv.org/pdf/2006.16236.pdf))
* code: https://github.com/idiap/fast-transformers


# Citation
If you found our research helpful or influential please consider citing:

BibTeX:
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

# Contact:
* Ádám Fodor (foauaai@inf.elte.hu)