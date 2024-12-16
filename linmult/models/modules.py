import torch
import torch.nn as nn
from linmult.models.transformer import TransformerEncoder, TemporalFactory


class TRM(nn.Module):
    """Time Reduce Module
    
    Time dimension is aggregated, transforms tensor with shape (B, T, F) to (B, F)
    """

    def __init__(self, config: dict):
        super().__init__()

        if config.get("time_dim_reducer") not in {None, 'attentionpool', 'gmp', 'gap', 'last'}:
            raise Exception(f'Invalid time_dim_reducer: {config.get("time_dim_reducer")}')

        self.time_dim_reducer = TemporalFactory.time_dim_reducer(config) # TR


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.time_dim_reducer(x, mask) # (B, T, F) -> (B, F)


    def apply_to_list(self, x_list: list[torch.Tensor], mask_list: list[torch.Tensor]) -> torch.Tensor:
        reduced_list = []
        for x, mask in zip(x_list, mask_list):
            reduced_x = self(x, mask) # (B, T, F) -> (B, F)
            reduced_list.append(reduced_x)
        return reduced_list


class TAM(nn.Module):
    """Time Align Module
    
    Time dimensions of multiple tensors are aligned, transforms tensors with shape (B, T, F) to (B, aligned_time_dim, F)
    """

    def __init__(self, config: dict):
        super().__init__()

        if config['time_dim_aligner'] not in {None, 'aap', 'amp', 'padding'}:
            raise Exception(f'Invalid time_dim_aligner: {config["time_dim_aligner"]}')
        
        self.aligned_time_dim = config['aligned_time_dim']
        self.time_dim_aligner = TemporalFactory.time_dim_aligner(config['time_dim_aligner']) # TA
        self.transformer = TransformerEncoder(config=config.copy() | {
            "d_model": config.get("src_dim"),
        })
        self.projector = nn.Conv1d(config.get("src_dim"), config.get("tgt_dim"), kernel_size=1, padding=0, bias=False)


    def forward(self, x_list: list[torch.Tensor], mask_list: list[torch.Tensor | None]) -> tuple[torch.Tensor, torch.Tensor]:
        x_new_list, mask_new_list = zip(*[
            self.time_dim_aligner(x=x, time_dim=self.aligned_time_dim, mask=mask)
            for x, mask in zip(x_list, mask_list)
        ])
        mm_x = torch.cat(x_new_list, dim=2) # (B, T, N*F)
        mask = torch.stack(mask_new_list, dim=0).any(dim=0) # (B, T)
        mm_x = self.transformer(mm_x, query_mask=mask) # (B, T, N*F)
        x = self.projector(mm_x.transpose(1, 2)).transpose(1, 2) # (B, T, N*F) -> (B, T, F)
        x = x * mask.unsqueeze(-1) # (B, T, F) * (B, T, 1)
        return x, mask # (B, T, F) and (B, T)