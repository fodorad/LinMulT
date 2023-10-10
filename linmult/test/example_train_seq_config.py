import torch
from linmult.models.LinMulT import LinMulT


model = LinMulT(input_modality_channels=[1024, 160],
                output_dim=1).cuda()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters())

for i in range(5):
    print(f'epoch: {i}')
    x_0 = torch.rand((8, 15, 1024), device='cuda')
    x_1 = torch.rand((8, 15, 160), device='cuda')
    y_true_cls = torch.rand((8, 1), device='cuda')
    y_true_seq = torch.rand((8, 15, 1), device='cuda')

    y_pred_cls, y_pred_seq = model([x_0, x_1])

    loss1 = criterion(y_pred_cls, y_true_cls)
    loss2 = criterion(y_pred_seq, y_true_seq)
    loss = loss1 + loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()