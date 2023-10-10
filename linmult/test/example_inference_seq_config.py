import torch
from linmult.models.LinMulT import LinMulT


x_1 = torch.rand((32, 15, 1024))
x_2 = torch.rand((32, 15, 160))
model = LinMulT(input_modality_channels=[1024, 160], output_dim=5)
y_pred_cls, y_pred_seq = model([x_1, x_2])

print(model)
print('y_pred_cls shape:', y_pred_cls.shape)
print('y_pred_seq shape:', y_pred_seq.shape)