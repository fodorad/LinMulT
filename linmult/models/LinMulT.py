import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from linmult.models.transformer import TransformerEncoder, TimeReduceFactory
from linmult.models.config_loader import load_config


class LinMulT(nn.Module):

    def __init__(self, config: dict | str):
        """Construct LinMulT: Linear-complexity Multimodal Transformer."""
        super().__init__()

        if isinstance(config, str):
            config = load_config(config)

        self.input_modality_channels = config.get("input_modality_channels") # [M_1, ..., M_N]
        self.output_dim = config.get("output_dim")
        self.n_modalities = len(self.input_modality_channels) # N
        self.d_model = config.get("d_model", 40)
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 6)
        self.dropout_embedding = config.get("dropout_embedding", 0.)
        self.dropout_ca = config.get("dropout_ca", 0.1)
        self.dropout_sa = config.get("dropout_sa", 0.1)
        self.dropout_relu = config.get("dropout_relu", 0.1)
        self.dropout_residual = config.get("dropout_residual", 0.1)
        self.dropout_output = config.get("dropout_output", 0.)

        self.add_time_collapse = False
        if config.get("time_reduce_type", None) is not None:
            self.time_reduce_module = TimeReduceFactory.create_time_reduce_layer(config)
            self.add_time_collapse = True

        self.time_reduce_type = config.get("time_reduce_type", None)
        if config.get("time_reduce_type", None) not in {None, 'attentionpool', 'gmp', 'gap', 'last'}:
            raise Exception(f'Invalid aggregation {self.time_reduce_type}.')

        self.aggregation = config.get("aggregation", None)
        if self.aggregation not in {None, 'last', 'cls', 'meanpooling', 'maxpooling'}:
            raise Exception(f'Invalid aggregation {self.aggregation}.')

        self.add_cm_attention_back = config.get("add_cm_attention_back", False)
        self.add_self_attention_fusion = config.get("add_self_attention_fusion", False)
        self.add_ffn_fusion = config.get("add_ffn_fusion", False)

        # 1. Temporal Convolutional Layers
        self.projectors = nn.ModuleList([
            nn.Conv1d(input_channels, self.d_model, kernel_size=1, padding=0, bias=False)
            for input_channels in self.input_modality_channels
        ])

        # 2. Crossmodal Attention Transformers
        # e.g.: a, v, t modalities correspond to 0, 1, 2 indices
        # Q -> a, K and V -> v, t:  v t - 1 2
        # Q -> v, K and V -> a, t:  a t - 0 2
        # Q -> t, K and V -> a, v:  a v - 0 1
        self.modality_indices = range(self.n_modalities)
        self.crossmodal_transformers = nn.ModuleList([])
        for target_index in self.modality_indices: # e.g. target_index = 0
            input_indices = [ind for ind in self.modality_indices if ind != target_index] # e.g. input_indices = [1, 2]
            self.crossmodal_transformers.append(
                nn.ModuleList([
                    TransformerEncoder(config=config)
                    for _ in input_indices
                ])
            )

        # 3. Branch-wise Self Attention Transformers
        self.self_attention_transformers = nn.ModuleList([
            TransformerEncoder(
                config=config.copy() | {
                    "d_model": (self.n_modalities - 1) * self.d_model,
                    "n_layers": config.get("n_layers_sa", 3)
                }
            )
            for _ in self.modality_indices # target_index
        ])

        # Optional: Crossmodal Attention Transformers Back
        if self.add_cm_attention_back:
            self.crossmodal_transformers_b = nn.ModuleList([
                TransformerEncoder(
                    config=config | {
                        "d_model": (self.n_modalities - 1) * self.d_model
                    }
                )
                for _ in range(self.n_modalities-1) # 0th target_index is ignored
            ])

        # Optional: Self Attention Fusion Transformer
        if self.add_self_attention_fusion:
            self.self_attention_fusion_transformer = TransformerEncoder(
                config=config.copy() | {
                    "d_model": self.d_model if self.add_cm_attention_back else self.n_modalities * self.d_model,
                    "n_layers": config.get("n_layers_sa_fusion", 3)
                }
            )

        if self.add_cm_attention_back and self.n_modalities == 2:
            combined_dim = self.d_model
        elif self.add_cm_attention_back and self.n_modalities > 2:
            combined_dim = (self.n_modalities - 1) * (self.n_modalities - 1) * self.d_model
        else:
            combined_dim = (self.n_modalities - 1) * self.n_modalities * self.d_model

        # Optional: FFN + Residual
        if self.add_ffn_fusion:
            self.projection_1 = nn.Linear(combined_dim, combined_dim)
            self.projection_2 = nn.Linear(combined_dim, combined_dim)

        # Sequence Head & Aggregation
        self.out_heads = nn.ModuleList([
            nn.Linear(combined_dim, output_dim)
            for output_dim in self.output_dim
        ]) # list of (B, T, output_dim) or (B, output_dim)


    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Inference with LinMulT.

        Args:
            inputs (list[torch.Tensor]): input tensors of shape (B, T, F)

        Returns:
            (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): tensor of shape (B, F) and/or (B, T, F)
        """
        # expand time dimension and repeat if feature vector is provided instead of a sequence
        # (B, F) -> (B, T_1, F)
        expand_time = lambda x: x if x.ndim == 3 else x.unsqueeze(1).expand(-1, inputs[0].shape[1], -1)
        inputs = [expand_time(inp) for inp in inputs]

        # temporal convolution projection of input tensors
        proj_x_mod = [self.projectors[i](
                         F.dropout(
                             input.transpose(1, 2),
                             p=self.dropout_embedding,
                             training=self.training)
                         ).transpose(1, 2)
                      for i, input in enumerate(inputs)] # (B, F, T) -> (B, T, d_model)
        logging.debug(f'projected input sizes: {[tuple(i.size()) for i in proj_x_mod]}')

        if self.aggregation == 'cls':
            # add cls token to every input as the first timestamp
            # (d_model,) -> (1, 1, d_model) -> (batch_size, 1, d_model)
            cls_tokens = [
                torch.zeros((proj_x_mod[i].shape[0], 1, proj_x_mod[i].shape[-1]), device=proj_x_mod[i].device)
                for _ in range(self.n_modalities)
            ]

            proj_x_mod = [
                torch.cat((cls_token, projected_representation), dim=1)
                for projected_representation, cls_token in zip(proj_x_mod, cls_tokens)
            ] # (B, T, d_model) -> (B, T+1, d_model)

        # cross-modal transformers
        hidden_representations = []
        for target_index in range(self.n_modalities): # e.g. target_index == 0
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

            # within branch self-attention transformer
            cross_modal_hidden = torch.cat(cross_modal_hidden, dim=2) # (B, T, n_modalities * d_model)
            self_hidden = self.self_attention_transformers[target_index](cross_modal_hidden)
            hidden_representations.append(self_hidden) # (B, T, F) or (B, T+1, F)
        logging.debug(f"last hidden representations with shapes: {[tuple(elem.size()) for elem in hidden_representations]}")

        # crossmodal transformers B
        if self.add_cm_attention_back:
            target_index = 0
            input_indices = [ind for ind in self.modality_indices if ind != target_index]
            target_crossmodal_hidden = []
            for i, input_index in enumerate(input_indices):
                target_crossmodal_hidden.append(
                    self.crossmodal_transformers_b[i](
                        hidden_representations[target_index], hidden_representations[input_index], hidden_representations[input_index])
                )
            hidden_representations = target_crossmodal_hidden
            logging.debug(f"num of crossmodal transformers B: {len(target_crossmodal_hidden)}, tensor shapes: {[tuple(elem.size()) for elem in target_crossmodal_hidden]}")

        if self.add_time_collapse:
            hidden_representation = torch.cat([self.time_reduce_module(hidden_representation) for hidden_representation in hidden_representations], dim=-1) # [(B, T, F), ...] -> (B, combined_dim)
        else:
            hidden_representation = torch.cat(hidden_representations, dim=-1) # [(B, T, F), ...] -> (B, T, combined_dim)
        logging.debug(f"branch concatenation shape: {hidden_representation.shape}")

        if self.add_self_attention_fusion:
            hidden_representation = self.self_attention_fusion_transformer(hidden_representation) # (B, T, combined_dim) -> (B, T, combined_dim)
            logging.debug(f"self attention fusion output shape: {hidden_representation.shape}")

        if self.add_ffn_fusion: # residual
            hidden_representation = self.projection_2(
                F.dropout(
                    F.relu(self.projection_1(hidden_representation)),
                    p=self.dropout_output, training=self.training
                )
            ) + hidden_representation # (B, T, combined_dim) or (B, combined_dim)
            logging.debug(f"projection fusion output shape: {hidden_representation.shape}")

        if self.add_time_collapse:
            output_cls = [out_layer(hidden_representation) for out_layer in self.out_heads] # for each head (B, combined_dim) -> (B, output_dim)
            logging.debug(f"time collapse output shape: {[elem.shape for elem in output_cls]}")
            return output_cls
        else:
            match self.aggregation:
                case 'last':
                    output_cls = [out_layer(hidden_representation[:, -1, :]) for out_layer in self.out_heads] # for each head (B, combined_dim)
                case 'cls':
                    output_cls = [out_layer(hidden_representation[:, 0, :]) for out_layer in self.out_heads] # for each head (B, T+1, combined_dim) -> (B, combined_dim) 
                    hidden_representation = hidden_representation[:, 1:, :] # (B, T+1, combined_dim) -> (B, T, combined_dim)
                case 'maxpooling':
                    output_cls = [torch.max(out_layer(hidden_representation), dim=1) for out_layer in self.out_heads] # for each head (B, T, combined_dim) -> (B, combined_dim)
                case _: # 'meanpooling'
                    output_cls = [torch.mean(out_layer(hidden_representation), dim=1) for out_layer in self.out_heads] # for each head (B, T, combined_dim) -> (B, combined_dim)

            # output_cls head: sequence -> aggregation -> dense -> summarized logits
            # output_seq head: sequence -> time-distributed dense -> sequence-wise logits
            output_seq = [out_layer(hidden_representation) for out_layer in self.out_heads]
            logging.debug(f"output output_cls shape: {[tuple(elem.size()) for elem in output_cls]}")
            logging.debug(f"output output_seq shape: {[tuple(elem.size()) for elem in output_seq]}")
            return output_cls, output_seq


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    batch_size = 8
    x_1 = torch.rand((batch_size, 450, 256))
    x_2 = torch.rand((batch_size, 1500, 25))

    config_model = load_config('configs/2M_S_2O_V_GAP_mha.yaml')
    #config_model = load_config('configs/2M_S_2O_S_CMB.yaml')

    model = LinMulT(config=config_model)

    output_cls = model([x_1, x_2])
    print('x_1:', x_1.shape, 'x_2:', x_2.shape, 'output_cls:', [output.shape for output in output_cls])

    # output_cls, output_seq = model([x_1, x_2])
    # print('x_1:', x_1.shape, 'x_2:', x_2.shape, 'output_cls:', [output.shape for output in output_cls], 'output_seq:', [output.shape for output in output_seq])