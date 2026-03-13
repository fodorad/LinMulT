# Configuration Reference

LinT uses {py:class}`~linmult.core.config.LinTConfig`
for single-modality models. LinMulT uses
{py:class}`~linmult.core.config.LinMulTConfig`
for multimodal models. Each output head is described by a
{py:class}`~linmult.core.config.HeadConfig`.

```python
from linmult import LinT, LinTConfig, LinMulT, LinMulTConfig, HeadConfig

# Single-modality (LinT)
cfg = LinTConfig(
    input_feature_dim=25,
    heads=[HeadConfig(type="simple", output_dim=3, name="emotion")],
)
model = LinT(cfg)

# Multimodal (LinMulT)
cfg = LinMulTConfig(
    input_feature_dim=[25, 35],
    heads=[HeadConfig(type="simple", output_dim=3, name="sentiment")],
)
model = LinMulT(cfg)

# From a dict (e.g. loaded from YAML)
cfg = LinTConfig.from_dict({"input_feature_dim": 25, ...})
cfg = LinMulTConfig.from_dict({"input_feature_dim": [25, 35], ...})

# From a YAML file
cfg = LinTConfig.from_yaml("lint.yaml")
cfg = LinMulTConfig.from_yaml("linmult.yaml")
```

---

## Attention variants

| `attention_type` | Algorithm | Complexity |
|-----------------|-----------|------------|
| `linear` | Linear attention (Katharopoulos et al., ICML 2020) | O(N·D²) |
| `performer` | FAVOR+ (Choromanski et al., ICLR 2021) | O(N·r·D) |
| `flash` | Gated Attention Unit (Hua et al., ICML 2022) | O(N·s) |
| `bigbird` | BigBird sparse attention | O(N·√N) |
| `softmax` | Scaled dot-product attention | O(N²) |
| `mha` | `nn.MultiheadAttention` | O(N²) |

---

## Example YAML (`LinMulT`)

```yaml
input_feature_dim: [25, 41, 768]

heads:
  - name: valence
    type: sequence_aggregation
    output_dim: 7
    norm: bn
    pooling: gap
  - name: arousal
    type: sequence
    output_dim: 2

d_model: 40
num_heads: 8
cmt_num_layers: 6
attention_type: linear

time_dim_reducer: gap

add_module_unimodal_sat: false
add_module_multimodal_signal: true
tam_aligner: amp
tam_time_dim: 300
mms_num_layers: 6

dropout_input: 0.0
dropout_pe: 0.0
dropout_ffn: 0.1
```

## Example YAML (`LinT`)

```yaml
input_feature_dim: 25

heads:
  - name: emotion
    type: sequence_aggregation
    output_dim: 7

d_model: 40
num_heads: 8
cmt_num_layers: 6
attention_type: linear

time_dim_reducer: attentionpool

dropout_input: 0.0
dropout_pe: 0.0
dropout_ffn: 0.1
```
