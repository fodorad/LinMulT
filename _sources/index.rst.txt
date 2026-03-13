LinMulT
=======

General-purpose Multimodal Transformer with Linear-Complexity Attention.

Handles variable-length inputs across any number of modalities, supports
missing-modality scenarios, and offers six attention variants from O(N²)
softmax to O(N·s) gated linear attention — all behind a single config key.

Installation
------------

.. code-block:: bash

   pip install linmult

Quick Start
-----------

Single modality (LinT):

.. code-block:: python

   import torch
   from linmult import LinT, LinTConfig

   x = torch.rand(8, 1500, 25)  # (batch, time, features)
   model = LinT(LinTConfig.from_dict({
       "input_feature_dim": 25,
       "heads": [{"name": "out", "type": "simple", "output_dim": 5}],
       "time_dim_reducer": "attentionpool",
   }))
   result = model(x)  # {"out": (8, 5)}

Multiple modalities (LinMulT):

.. code-block:: python

   import torch
   from linmult import LinMulT, LinMulTConfig

   x1, x2 = torch.rand(8, 1500, 25), torch.rand(8, 450, 35)
   model = LinMulT(LinMulTConfig.from_dict({
       "input_feature_dim": [25, 35],
       "heads": [{"name": "sentiment", "type": "simple", "output_dim": 3}],
       "time_dim_reducer": "gap",
   }))
   result = model([x1, x2])  # {"sentiment": (8, 3)}

.. toctree::
   :maxdepth: 2
   :caption: Guides

   config-reference
