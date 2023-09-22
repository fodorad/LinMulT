import torch
from linmult.models.LinMulT import LinMulT


x_a = torch.rand((32, 1500, 35))
x_v = torch.rand((32, 450, 25))
x_t = torch.rand((32, 105, 768))
model = LinMulT(input_modality_channels=[35, 25, 768],
                output_dim=5,
                add_time_collapse=True,
                add_self_attention_fusion=False)
y_pred = model([x_a, x_v, x_t])
print('y_pred shape:', y_pred.shape)