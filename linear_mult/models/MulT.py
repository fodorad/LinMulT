import os
import subprocess
from absl import logging
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from linear_mult.models.transformer import TransformerEncoder


logging.set_verbosity(logging.INFO) # DEBUG


# supported pretrained weights from transformers paper
PRETRAINED_WEIGHTS = {
    'fi-linmult-oob-wfr':   'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr.ckpt',   # MT
    'fi-linmult-oob-wfr-0': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr-0.ckpt', # TW O
    'fi-linmult-oob-wfr-1': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr-1.ckpt', # TW C
    'fi-linmult-oob-wfr-2': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr-2.ckpt', # TW E
    'fi-linmult-oob-wfr-3': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr-3.ckpt', # TW A
    'fi-linmult-oob-wfr-4': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-wfr-4.ckpt', # TW N
    'fi-linmult-oob':       'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob.ckpt',       # MT
    'fi-linmult-oob-0':     'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-0.ckpt',     # TW O
    'fi-linmult-oob-1':     'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-1.ckpt',     # TW C
    'fi-linmult-oob-2':     'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-2.ckpt',     # TW E
    'fi-linmult-oob-3':     'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-3.ckpt',     # TW A
    'fi-linmult-oob-4':     'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-4.ckpt',     # TW N
    'fi-linmult-oob-0-old': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-0-old.ckpt', # TW O
    'fi-linmult-oob-1-old': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-1-old.ckpt', # TW C
    'fi-linmult-oob-2-old': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-2-old.ckpt', # TW E
    'fi-linmult-oob-3-old': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-3-old.ckpt', # TW A
    'fi-linmult-oob-4-old': 'nipg12.inf.elte.hu/~fodor@nipg.lab/pretrained_weights/fi-linmult-oob-4-old.ckpt', # TW N
}


class MulT(nn.Module):

    def __init__(self, input_modality_channels: list,
                       output_dim: int,
                       only_target_branch: list = None,
                       projected_modality_dim: int | list = None,
                       num_heads: int = 8,
                       layers: int = 5,
                       attn_dropout: float = 0.0,
                       attn_dropout_mod: list = None,
                       relu_dropout: float = 0.2,
                       res_dropout: float = 0.2,
                       embed_dropout_mod: list = None,
                       out_dropout: float = 0.2,
                       attn_mask: bool = True,
                       attention_type: str = 'linear',
                       weights: str = None):
        """
        Construct a MulT model with linear or softmax attention.
        """
        super().__init__()

        assert attention_type in {'linear', 'softmax'}

        self.input_modality_channels = input_modality_channels # [tuple,...]
        self.number_of_modalities = len(self.input_modality_channels)

        if projected_modality_dim is None:
            self.projected_modality_dim = [32] * self.number_of_modalities
        elif isinstance(projected_modality_dim, list):
            assert len(projected_modality_dim) == self.number_of_modalities
            self.projected_modality_dim = projected_modality_dim
        elif isinstance(projected_modality_dim, int):
            self.projected_modality_dim = [projected_modality_dim] * self.number_of_modalities
        else:
            raise ValueError(f'Invalid projected modality argument: {projected_modality_dim}')

        self.only_target_branch = only_target_branch if only_target_branch is not None else [True] * self.number_of_modalities # [bool,...]
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.attn_dropout_mod = attn_dropout_mod if attn_dropout_mod is not None else [0.1] * self.number_of_modalities # [float,...]
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout_mod = embed_dropout_mod if embed_dropout_mod is not None else [0.1] * self.number_of_modalities
        self.attn_mask = attn_mask
        self.attention_type = attention_type

        self.partial_mode = torch.tensor(self.only_target_branch).sum()
        combined_dim = (self.number_of_modalities-1) * torch.tensor(self.projected_modality_dim).sum() # 2 * 3 * 30 = 180 # source_mod, num_mod, projected_modality_dim

        # 1. Temporal convolutional layers
        self.projectors = nn.ModuleList([nn.Conv1d(input_modality_channels, projected_modality_dim, kernel_size=1, padding=0, bias=False)
                                         for input_modality_channels, projected_modality_dim in zip(self.input_modality_channels, self.projected_modality_dim)])

        # 2. Crossmodal Attentions
        # e.g.:   a v t - 0 1 2
        # only_a:   v t - 1 2
        # only_v:   a t - 0 2
        # only_t:   a v - 0 1
        self.modality_indices = range(self.number_of_modalities)
        self.crossmodal_transformers = nn.ModuleList([])
        for target_index in self.modality_indices: # e.g. target_index = 0
            input_indices = [ind for ind in self.modality_indices if ind != target_index] # e.g. input_indices = [1, 2]
            self.crossmodal_transformers.append(
                nn.ModuleList([self.create_transformer(modality_index=input_index, attention_type='cross')
                               for input_index in input_indices])
            )

        # 3. Self Attentions
        self.self_attention_transformers = nn.ModuleList([self.create_transformer(modality_index=target_index, attention_type='self', layers=3)
                                                          for target_index in self.modality_indices])

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        if weights is not None:
            self.load_weights(weights)


    def load_weights(self, weights_path: str | Path):
        weights_path = Path(weights_path)

        if weights_path.name in PRETRAINED_WEIGHTS.keys():
            default_weights_location = Path().home() / '.cache' / 'torch' / 'hub' / 'checkpoints' / 'MulT'
            key = weights_path.name
            weights_path = default_weights_location / Path(PRETRAINED_WEIGHTS[key]).name

            if not (weights_path).exists():
                logging.log(logging.INFO, 'Check server availablility...')
                assert '200' in subprocess.check_output(f"curl -I --http2 {PRETRAINED_WEIGHTS[key]}", shell=True).decode('UTF-8').split('\n')[0], \
                    f'Server is not available or missing weights at {PRETRAINED_WEIGHTS[key]}'
                default_weights_location.mkdir(parents=True, exist_ok=True)
                logging.log(logging.INFO, f'Downloading {key} weights to {weights_path}')
                os.system(f'curl {PRETRAINED_WEIGHTS[key]} --output {weights_path}')

            assert weights_path.exists(), f'Pretrained weights are missing at {str(weights_path)}'

        state_dict = torch.load(str(weights_path))['state_dict']
        # If the model was initialized with nn.DataParallel, the saved parameter names have a prefix: 'model.'
        state_dict = {k.partition('model.')[2]:state_dict[k] for k in state_dict.keys()}
        self.load_state_dict(state_dict)
        logging.log(logging.INFO, f'MulT weights are loaded from {weights_path}')


    def create_transformer(self, modality_index, attention_type: str, layers=-1):
        assert attention_type in {'cross', 'self'}

        if attention_type == 'cross': # cross-modal attention transformer
            embed_dim, attn_dropout = self.projected_modality_dim[modality_index], self.attn_dropout_mod[modality_index]
        else: # self-attention transformer
            embed_dim, attn_dropout = (self.number_of_modalities-1)*self.projected_modality_dim[modality_index], self.attn_dropout

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout_mod[modality_index],
                                  attn_mask=self.attn_mask,
                                  attention_type=self.attention_type)


    def forward(self, inputs: list):
        """
        input tensors should have dimension [batch_size, seq_len, n_features]
        """
        inp = [] # x_a, x_v, x_t
        for i, input in enumerate(inputs):
            if self.embed_dropout_mod[i] > 0:
                inp.append(F.dropout(input.transpose(1, 2), p=self.embed_dropout_mod[i], training=self.training))
            else:
                inp.append(input.transpose(1, 2))

        # projection of input tensors
        logging.log(logging.DEBUG, f'input sizes: {[tuple(i.size()) for i in inp]}')
        proj_x_mod = []
        for i, input in enumerate(inp):
            proj_x = input if self.input_modality_channels == self.projected_modality_dim else self.projectors[i](input)
            proj_x = proj_x.permute(0, 2, 1)
            proj_x_mod.append(proj_x)
        logging.log(logging.DEBUG, f'projected input sizes: {[tuple(i.size()) for i in proj_x_mod]}')

        # cross-modal transformers
        last_hidden = []
        for target_index in range(self.number_of_modalities):
            if self.only_target_branch[target_index]:
                input_indices = [ind for ind in self.modality_indices if ind != target_index] # e.g. t_mods = [1, 2]
                cross_modal_hidden = []
                for i, input_index in enumerate(input_indices):
                    # AVT: (V,T) --> A
                    logging.log(logging.DEBUG, f"Query: {[f'modality_{m}' for m in self.modality_indices][target_index]} with shape {tuple(proj_x_mod[target_index].size())} " + \
                                               f"--> Keys, Values: {[f'modality_{m}' for m in self.modality_indices][input_index]} with shape {tuple(proj_x_mod[input_index].size())}")
                    cross_modal_hidden.append(self.crossmodal_transformers[target_index][i](proj_x_mod[target_index], proj_x_mod[input_index], proj_x_mod[input_index])) # Q, K, V
                logging.log(logging.DEBUG, f"num of crossmodal transformers: {len(cross_modal_hidden)}, tensor shapes: {[tuple(elem.size()) for elem in cross_modal_hidden]}")

                # self-attention transformer
                cross_modal_hidden = torch.cat(cross_modal_hidden, dim=2) # concatenate if the target is the same
                self_hidden = self.self_attention_transformers[target_index](cross_modal_hidden)
                last_hidden.append(self_hidden[:,-1,:]) # Take the last output for prediction
        logging.log(logging.DEBUG, f"last hidden representations with shapes: {[tuple(elem.size()) for elem in last_hidden]}")

        # residual connection
        shortcut = torch.cat(last_hidden, dim=1)
        last_hidden_representation = self.proj2(F.dropout(F.relu(self.proj1(shortcut)), p=self.out_dropout, training=self.training))
        last_hidden_representation += shortcut

        # linear dense
        output = self.out_layer(last_hidden_representation)
        logging.log(logging.DEBUG, f"output shape: {tuple(output.size())}")
        return output


if __name__ == "__main__":

    i2 = [torch.rand((1, 5, 4)),
          torch.rand((1, 6, 5))]

    i3 = [torch.rand((1, 5, 4)),
          torch.rand((1, 6, 5)),
          torch.rand((1, 7, 6))]

    i4 = [torch.rand((1, 5, 4)),
          torch.rand((1, 7, 5)),
          torch.rand((1, 6, 6)),
          torch.rand((1, 8, 7))]

    i5 = [torch.rand((1, 4, 4)),
          torch.rand((1, 5, 5)),
          torch.rand((1, 6, 6)),
          torch.rand((1, 7, 7)),
          torch.rand((1, 8, 8))]

    for num_modality, inputs in zip([2, 3, 4, 5], [i2, i3, i4, i5]):
        model = MulT([i.shape[-1] for i in inputs], 1)
        output = model(inputs)
        print(f'Number of modality: {num_modality}, model output shape: {output.size()}')

    # simulate opensmile egemaps, openface action units, wav2vec, fabnet, roberta features as inputs
    opensmile_egemaps = torch.rand((10, 1500, 25))
    openface_action_units = torch.rand((10, 450, 35))
    wav2vec = torch.rand((10, 1500, 512))
    fabnet = torch.rand((10, 450, 256))
    roberta = torch.rand((10, 105, 1024))
    model = MulT((25, 35, 512, 256, 1024), 5)
    output = model([opensmile_egemaps, openface_action_units, wav2vec, fabnet, roberta])
    print(f'Model with 5 modalities, expected shape is (B, 5): {output.size()}')

    model = MulT((25, 35, 512, 256, 1024), 1, weights='fi-linmult-oob-wfr-0')
    print(f'Model is loaded with pretrained weights.')

