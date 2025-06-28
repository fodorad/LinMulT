# LinMulT

[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

General-purpose Multimodal Transformer with Linear-Complexity Attention Mechanism.

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

The following use cases demonstrate some basic, then more advanced functionality using the LinT and LinMulT models. For better coverage of configurations, refer to the test cases provided in the **linmult/test** directory.

## LinT: linear-complexity transformer for a single input sequence

### Input sequence without mask, single output head

```
import torch
from linmult import LinT

batch_size = 8
time_dim_1 = 1500
feature_dim_1 = 25
output_dim_1 = 5

x = torch.rand((batch_size, time_dim_1, feature_dim_1))

model = LinT(
    {
        'input_feature_dim': feature_dim_1,
        'heads': [{'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1}]
    }
)
output_heads = model(x)
assert output_heads['head_0'].shape == (batch_size, time_dim_1, output_dim_1)
```

### Input sequence without mask, single aggregated output head

Note, that the time dimension aggregation is applied within the model.

```
import torch
from linmult import LinT

batch_size = 8
time_dim_1 = 1500
feature_dim_1 = 25
output_dim_1 = 5

x = torch.rand((batch_size, time_dim_1, feature_dim_1))

model = LinT(
    {
        'input_feature_dim': feature_dim_1,
        'heads': [{'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1}],
        'time_dim_reducer': 'attentionpool'
    }
)
output_heads = model(x)
assert output_heads['head_0'].shape == (batch_size, output_dim_1)
```

### Input sequence without mask, single aggregated output head

Note, that the time dimension aggregation is applied to the model outputs.

```
import torch
from linmult import LinT, apply_logit_aggregation

batch_size = 8
time_dim_1 = 1500
feature_dim_1 = 25
output_dim_1 = 5

x = torch.rand((batch_size, time_dim_1, feature_dim_1))

model = LinT(
    {
        'input_feature_dim': feature_dim_1,
        'heads': [{'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1}]
    }
)
output_heads = model(x)
assert output_heads['head_0'].shape == (batch_size, time_dim_1, output_dim_1)

output = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
assert output.shape == (batch_size, output_dim_1)
```

### Input sequence with a mask, multiple aggregated output head

```
import torch
from linmult import LinT, apply_logit_aggregation

batch_size = 8
time_dim_1 = 50
feature_dim_1 = 25
output_dim_1 = 5
output_dim_2 = 6

x = torch.rand((batch_size, time_dim_1, feature_dim_1))
mask = (torch.arange(x.size(1)).unsqueeze(0) < x.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_1) 

model = LinT(
    {
        'input_feature_dim': feature_dim_1,
        'heads': [
            {'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1},
            {'name': 'head_1', 'type': 'simple', 'output_dim': output_dim_2}
        ]
    }
)
output_heads = model(x, mask)
assert output_heads['head_0'].shape == (batch_size, time_dim_1, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, time_dim_1, output_dim_2)

output_0 = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
output_1 = apply_logit_aggregation(x=output_heads['head_1'], method='meanpooling')
assert output_0.shape == (batch_size, output_dim_1)
assert output_1.shape == (batch_size, output_dim_2)
```

### Input sequence with a mask, a sequence and an aggregated output head 

```
import torch
from linmult import LinT, apply_logit_aggregation

batch_size = 8
time_dim_1 = 50
feature_dim_1 = 25
output_dim_1 = 5
output_dim_2 = 6

x = torch.rand((batch_size, time_dim_1, feature_dim_1))
mask = (torch.arange(x.size(1)).unsqueeze(0) < x.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_1) 

model = LinT(
    {
        'input_feature_dim': feature_dim_1,
        'heads': [
            {'name': 'head_0', 'type': 'sequence', 'output_dim': output_dim_1},
            {'name': 'head_1', 'type': 'sequence_aggregation', 'output_dim': output_dim_2}
        ]
    }
)
output_heads = model(x, mask)
assert output_heads['head_0'].shape == (batch_size, time_dim_1, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, output_dim_2)

output_0 = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
assert output_0.shape == (batch_size, output_dim_1)
```

## LinMulT: linear-complexity multimodal transformer for multiple input sequences

### 2 input sequences with same time dimensions, single aggregated output

```
import torch
from linmult import LinMulT, apply_logit_aggregation

batch_size = 8
time_dim = 450
feature_dim_1, feature_dim_2 = 25, 35
output_dim_1 = 5

x_1 = torch.rand((batch_size, time_dim, feature_dim_1))
x_2 = torch.rand((batch_size, time_dim, feature_dim_2))

model = LinMulT(
    {
        'input_feature_dim': [feature_dim_1, feature_dim_2],
        'heads': [{'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1}]
    }
)
output_heads = model([x_1, x_2])
assert output_heads['head_0'].shape == (batch_size, time_dim, output_dim_1)

output = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
assert output.shape == (batch_size, output_dim_1)
```

### 3 input sequences with different time dimensions, masks, multiple aggregated output heads

```
import torch
from linmult import LinMulT

batch_size = 8
time_dim_1, time_dim_2, time_dim_3 = 1500, 450, 450
feature_dim_1, feature_dim_2, feature_dim_3 = 25, 35, 256
output_dim_1 = 5
output_dim_2 = 6

x_1 = torch.rand((batch_size, time_dim_1, feature_dim_1))
x_2 = torch.rand((batch_size, time_dim_2, feature_dim_2))
x_3 = torch.rand((batch_size, time_dim_3, feature_dim_3))
mask_1 = (torch.arange(x_1.size(1)).unsqueeze(0) < x_1.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_1) 
mask_2 = (torch.arange(x_2.size(1)).unsqueeze(0) < x_2.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_2) 
mask_3 = (torch.arange(x_3.size(1)).unsqueeze(0) < x_3.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_3) 

model = LinMulT(
    {
        'input_feature_dim': [feature_dim_1, feature_dim_2, feature_dim_3],
        'heads': [
            {'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1},
            {'name': 'head_1', 'type': 'simple', 'output_dim': output_dim_2}
        ],
        'time_dim_reducer': 'gap',
    }
)
output_heads = model([x_1, x_2, x_3], [mask_1, mask_2, mask_3])
assert output_heads['head_0'].shape == (batch_size, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, output_dim_2)
```

### 3 input sequences with different time dimensions, a missing input, enhanced multimodal signal module

```
import torch
from linmult import LinMulT, apply_logit_aggregation

batch_size = 8
time_dim_1, time_dim_2, time_dim_3 = 1500, 450, 450
feature_dim_1, feature_dim_2, feature_dim_3 = 25, 35, 256
output_dim_1 = 5

x_1 = torch.rand((batch_size, time_dim_1, feature_dim_1))
x_2 = torch.rand((batch_size, time_dim_2, feature_dim_2))
x_3 = torch.rand((batch_size, time_dim_3, feature_dim_3))
mask_1 = (torch.arange(x_1.size(1)).unsqueeze(0) < x_1.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_1) 
mask_2 = (torch.arange(x_2.size(1)).unsqueeze(0) < x_2.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_2) 
mask_3f = torch.zeros(size=x_3.size()[:2], dtype=bool) # Shape: (B, T_3)

model = LinMulT(
    {
        'input_feature_dim': [feature_dim_1, feature_dim_2, feature_dim_3],
        'heads': [
            {'name': 'head_0', 'type': 'simple', 'output_dim': output_dim_1},
            {'name': 'head_1', 'type': 'simple', 'output_dim': output_dim_2}
        ],
        'multimodal_signal': True,
        'time_dim_aligner': 'amp',
        'tam_fusion': True,
        'aligned_time_dim': time_dim_2,
    }
)
output_heads = model([x_1, x_2, x_3], [mask_1, mask_2, mask_3f])
assert output_heads['head_0'].shape == (batch_size, time_dim_2, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, time_dim_2, output_dim_2)

output = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
assert output.shape == (batch_size, output_dim_1)
```


### 3 input sequences with different time dimensions, a missing input, enhanced multimodal signal module with multiple task-specific (sequence and aggregated sequence) output heads

```
import torch
from linmult import LinMulT, apply_logit_aggregation

batch_size = 8
time_dim_1, time_dim_2, time_dim_3 = 1500, 450, 450
feature_dim_1, feature_dim_2, feature_dim_3 = 25, 35, 256
output_dim_1 = 5

x_1 = torch.rand((batch_size, time_dim_1, feature_dim_1))
x_2 = torch.rand((batch_size, time_dim_2, feature_dim_2))
x_3 = torch.rand((batch_size, time_dim_3, feature_dim_3))
mask_1 = (torch.arange(x_1.size(1)).unsqueeze(0) < x_1.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_1) 
mask_2 = (torch.arange(x_2.size(1)).unsqueeze(0) < x_2.size(1) - 10).expand(batch_size, -1).bool() # Shape: (batch_size, time_dim_2) 
mask_3f = torch.zeros(size=x_3.size()[:2], dtype=bool) # Shape: (B, T_3)

model = LinMulT(
    {
        'input_feature_dim': [feature_dim_1, feature_dim_2, feature_dim_3],
        'heads': [
            {'name': 'head_0', 'type': 'sequence', 'output_dim': output_dim_1},
            {'name': 'head_1', 'type': 'sequence_aggregation', 'output_dim': output_dim_2}
        ],
        'multimodal_signal': True,
        'time_dim_aligner': 'amp',
        'tam_fusion': True,
        'aligned_time_dim': time_dim_2,
    }
)
output_heads = model([x_1, x_2, x_3], [mask_1, mask_2, mask_3f])
assert output_heads['head_0'].shape == (batch_size, time_dim_2, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, output_dim_2)

output = apply_logit_aggregation(x=output_heads['head_0'], method='meanpooling')
assert output.shape == (batch_size, output_dim_1)
```

### Using a config file

```
import torch
from linmult import LinMulT, apply_logit_aggregation, load_config

batch_size = 8
time_dim_1, time_dim_2, time_dim_3 = 300, 300, 500
feature_dim_1, feature_dim_2, feature_dim_3 = 25, 41, 768
output_dim_1 = 7
output_dim_2 = 2

x_1 = torch.rand((batch_size, time_dim_1, feature_dim_1))
x_2 = torch.rand((batch_size, time_dim_2, feature_dim_2))
x_3 = torch.rand((batch_size, time_dim_3, feature_dim_3))

config = load_config("configs/LinMulT.yaml")
model = LinMulT(config)

output_heads = model([x_1, x_2, x_3])
assert output_heads['head_0'].shape == (batch_size, output_dim_1)
assert output_heads['head_1'].shape == (batch_size, time_dim_1, output_dim_2)

output = apply_logit_aggregation(x=output_heads['head_1'], method='meanpooling')
assert output.shape == (batch_size, output_dim_2)
```

# Similar projects using LinMulT

### (2023) BlinkLinMulT

LinMulT is trained for blink presence detection and eye state recognition tasks.
Our results demonstrate comparable or superior performance compared to state-of-the-art models on 2 tasks, using 7 public benchmark databases.

* paper: BlinkLinMulT: Transformer-based Eye Blink Detection ([pdf](https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf), [website](https://www.mdpi.com/2313-433X/9/10/196))
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

* Ádám Fodor (fodorad201@gmail.com) [[website](https://adamfodor.com)]