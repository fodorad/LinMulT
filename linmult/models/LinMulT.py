##########################################################
#                                                        #
#   Code is inspired by the following repositories:      #
#   https://github.com/yaohungt/Multimodal-Transformer   #
#                                                        #
##########################################################
import logging
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")


class LinMulT(nn.Module):

    def __init__(self,
                 input_modality_channels: Iterable[int],
                 output_dim: int,
                 projected_modality_dim: int | list = 40, # d
                 number_of_heads: int = 8,
                 number_of_layers: int = 4, # D
                 embedding_dropout: float = 0.1,
                 cross_attention_dropout: float = 0.1,
                 self_attention_dropout: float = 0.0,
                 relu_dropout: float = 0.1,
                 residual_dropout: float = 0.1,
                 output_dropout: float = 0.1,
                 attention_mask: bool = True,
                 add_time_collapse: bool = False,
                 add_self_attention_fusion: bool = True,
                 add_projection_fusion: bool = True,
                 aggregation: str = 'meanpooling'):
        """Construct a MulT model with linear attention mechanism.

        Args:
            aggregation (str | None): aggregation applied to the output sequence to get output_cls.
                None - when add_time_collapse is True, aggregation is not used at all.
                last - last timestep is used. Original MulT implementation.
                cls - classification token is used.
                meanpooling - mean is calculated over the T time dimension.
                maxpooling - max is calculated over the T time dimension.
        """
        super().__init__()

        if aggregation not in {None, 'last', 'cls', 'meanpooling', 'maxpooling'}:
            raise Exception(f'Invalid aggregation {aggregation}.')

        if add_time_collapse and add_self_attention_fusion:
            raise Exception(f'These arguments cannot be True at the same time: {{add_time_collapse, add_self_attention_fusion}}')

        self.input_modality_channels = input_modality_channels
        self.output_dim = output_dim
        self.number_of_modalities = len(self.input_modality_channels)

        if isinstance(projected_modality_dim, int):
            self.projected_modality_dim = [projected_modality_dim] * self.number_of_modalities
        else: # list
            if len(projected_modality_dim) != self.number_of_modalities:
                raise Exception('Length of projected_modality_dim should be the number of modalities.')
            self.projected_modality_dim = projected_modality_dim

        self.number_of_heads = number_of_heads
        self.number_of_layers = number_of_layers
        self.embedding_dropout = embedding_dropout
        self.cross_attention_dropout = cross_attention_dropout
        self.self_attention_dropout = self_attention_dropout
        self.relu_dropout = relu_dropout
        self.residual_dropout = residual_dropout
        self.output_dropout = output_dropout
        self.attention_mask = attention_mask
        self.add_time_collapse = add_time_collapse
        self.add_self_attention_fusion = add_self_attention_fusion
        self.add_projection_fusion = add_projection_fusion
        self.aggregation = aggregation if not add_time_collapse else None
        combined_dim = (self.number_of_modalities - 1) * torch.tensor(self.projected_modality_dim).sum()

        # 1. Temporal Convolutional Layers
        self.projectors = nn.ModuleList([
            nn.Conv1d(input_modality_channels, projected_modality_dim, kernel_size=1, padding=0, bias=False)
            for input_modality_channels, projected_modality_dim
            in zip(self.input_modality_channels, self.projected_modality_dim)
        ])

        # 2. Crossmodal Attention Transformers
        # e.g.: a, v, t modalities correspond to 0, 1, 2 indices
        # Q -> a, K and V -> v, t:  v t - 1 2
        # Q -> v, K and V -> a, t:  a t - 0 2
        # Q -> t, K and V -> a, v:  a v - 0 1
        self.modality_indices = range(self.number_of_modalities)
        self.crossmodal_transformers = nn.ModuleList([])
        for target_index in self.modality_indices: # e.g. target_index = 0
            input_indices = [ind for ind in self.modality_indices if ind != target_index] # e.g. input_indices = [1, 2]
            self.crossmodal_transformers.append(
                nn.ModuleList([
                    self.create_transformer(modality_index=input_index, attention_type='cross')
                    for input_index in input_indices
                ])
            )

        # 3. Self Attention Transformers
        self.self_attention_transformers = nn.ModuleList([
            self.create_transformer(modality_index=target_index, attention_type='self', layers=3)
            for target_index in self.modality_indices
        ])

        # 4. Self Attention Fusion Transformer
        if self.add_self_attention_fusion:
            self.self_attention_fusion_transformer = self.create_fusion_transformer()

        if self.add_projection_fusion:
            self.projection_1 = nn.Linear(combined_dim, combined_dim)
            self.projection_2 = nn.Linear(combined_dim, combined_dim)

        # 5. Sequence Head & Aggregation
        self.out_layer = nn.Linear(combined_dim, self.output_dim) # (B, T, output_dim) or (B, output_dim)

    def create_transformer(self, modality_index, attention_type: str, layers=-1):
        if attention_type == 'cross': # Crossmodal Attention Transformer
            embedding_dim = self.projected_modality_dim[modality_index]
            attention_dropout = self.cross_attention_dropout
        else: # Self Attention Transformer
            embedding_dim = (self.number_of_modalities - 1) * self.projected_modality_dim[modality_index]
            attention_dropout = self.self_attention_dropout

        return TransformerEncoder(embedding_dim=embedding_dim,
                                  number_of_heads=self.number_of_heads,
                                  number_of_layers=max(self.number_of_layers, layers),
                                  attention_dropout=attention_dropout,
                                  relu_dropout=self.relu_dropout,
                                  residual_dropout=self.residual_dropout,
                                  embedding_dropout=self.embedding_dropout,
                                  attention_mask=self.attention_mask)

    def create_fusion_transformer(self, layers=-1):
        return TransformerEncoder(embedding_dim=self.number_of_modalities * self.projected_modality_dim[0],
                                  number_of_heads=self.number_of_heads,
                                  number_of_layers=max(self.number_of_layers, layers),
                                  attention_dropout=self.self_attention_dropout,
                                  relu_dropout=self.relu_dropout,
                                  residual_dropout=self.residual_dropout,
                                  embedding_dropout=self.self_attention_dropout,
                                  attention_mask=self.attention_mask)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference with Multimodal Transformer.

        Args:
            inputs (list[torch.Tensor]): input tensors of shape (B, T, F)

        Returns:
            (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): tensor of shape (B, F) and/or (B, T, F)
        """
        # transpose and add embedding dropout
        inp = []  # x_a, x_v, x_t
        for input in inputs:
            input_T = input.transpose(1, 2) # (B, T, F) -> (B, F, T)
            if self.embedding_dropout > 0:
                inp.append(F.dropout(input_T, p=self.embedding_dropout, training=self.training))
            else:
                inp.append(input_T)
        logging.debug(f'input sizes: {[tuple(i.size()) for i in inp]}')

        # temporal convolution projection of input tensors
        proj_x_mod = [self.projectors[i](input).permute(0, 2, 1) for i, input in enumerate(inp)]
        logging.debug(f'projected input sizes: {[tuple(i.size()) for i in proj_x_mod]}')

        if self.aggregation == 'cls':
            # add cls token to every input as the first timestamp
            # (projected_dim,) -> (1, 1, projected_dim) -> (batch_size, 1, projected_dim)
            cls_tokens = [
                torch.zeros((proj_x_mod[i].shape[0], 1, proj_x_mod[i].shape[-1]), device=proj_x_mod[i].device)
                for _ in range(self.number_of_modalities)
            ]

            proj_x_mod = [
                torch.cat((cls_token, projected_representation), dim=1)
                for projected_representation, cls_token in zip(proj_x_mod, cls_tokens)
            ] # (B, T, F) -> (B, T+1, F)

        # cross-modal transformers
        hidden_representations = []
        for target_index in range(self.number_of_modalities): # e.g. target_index == 0
            input_indices = [ind for ind in self.modality_indices if ind != target_index]  # e.g. input_indices = [1, 2]
            cross_modal_hidden = []
            for i, input_index in enumerate(input_indices):
                # AVT: (V,T) --> A
                logging.debug(f"Query: {[f'modality_{m}' for m in self.modality_indices][target_index]} with shape {tuple(proj_x_mod[target_index].size())} " + \
                              f"--> Keys, Values: {[f'modality_{m}' for m in self.modality_indices][input_index]} with shape {tuple(proj_x_mod[input_index].size())}")
                cross_modal_hidden.append(
                    self.crossmodal_transformers[target_index][i](
                        proj_x_mod[target_index], proj_x_mod[input_index], proj_x_mod[input_index])
                ) # Q, K, V
            logging.debug(f"num of crossmodal transformers: {len(cross_modal_hidden)}, tensor shapes: {[tuple(elem.size()) for elem in cross_modal_hidden]}")

            # self-attention transformer
            cross_modal_hidden = torch.cat(cross_modal_hidden, dim=2) # within branch
            self_hidden = self.self_attention_transformers[target_index](cross_modal_hidden)
            hidden_representations.append(self_hidden) # (B, T, F) or (B, T+1, F)
        logging.debug(f"last hidden representations with shapes: {[tuple(elem.size()) for elem in hidden_representations]}")

        if self.add_time_collapse:
            hidden_representation = torch.cat([hidden_representation[:,-1,:] for hidden_representation in hidden_representations], dim=-1) # [(B, T, F), ...] -> (B, combined_dim)
        else:
            hidden_representation = torch.cat(hidden_representations, dim=-1) # [(B, T, F), ...] -> (B, T, combined_dim)

        if self.add_self_attention_fusion:
            hidden_representation = self.self_attention_fusion_transformer(hidden_representation)

        if self.add_projection_fusion:
            hidden_representation = self.projection_2(F.dropout(F.relu(self.projection_1(hidden_representation)), p=self.output_dropout, training=self.training)) \
                + hidden_representation # (B, T, combined_dim) or (B, combined_dim)

        if self.add_time_collapse:
            output_cls = self.out_layer(hidden_representation)
            return output_cls
        else:
            match self.aggregation:
                case 'last':
                    output_cls = self.out_layer(hidden_representation[:, -1, :]) # (B, combined_dim)
                case 'cls':
                    output_cls = self.out_layer(hidden_representation[:, 0, :]) # (B, T+1, combined_dim) -> (B, combined_dim)
                    hidden_representation = hidden_representation[:, 1:, :] # (B, T+1, combined_dim) -> (B, T, combined_dim)
                case 'maxpooling':
                    output_cls = self.out_layer(torch.max(hidden_representation, dim=1)) # (B, T, combined_dim) -> (B, combined_dim)
                case _: # 'meanpooling'
                    output_cls = self.out_layer(torch.mean(hidden_representation, dim=1)) # (B, T, combined_dim) -> (B, combined_dim)

            # output_cls head: sequence -> aggregation -> dense -> summarized logits
            # output_seq head: sequence -> time-distributed dense -> sequence-wise logits
            output_seq = self.out_layer(hidden_representation)
            logging.debug(f"output output_cls shape: {tuple(output_cls.size())}")
            logging.debug(f"output output_seq shape: {tuple(output_seq.size())}")
            return output_cls, output_seq