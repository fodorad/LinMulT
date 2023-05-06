import torch
torch.cuda.set_device(0)

from linear_mult.models.MulT import MulT


model = MulT([35, 25, 512, 256, 1024], 5, add_cls_token=False, target_sequence=False).cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

for i in range(5):
    print(f'epoch: {i}')

    x_o1 = torch.rand((8, 1500, 35), device='cuda')
    x_o2 = torch.rand((8, 450, 25), device='cuda')
    x_w = torch.rand((8, 1500, 512), device='cuda')
    x_f = torch.rand((8, 450, 256), device='cuda')
    x_r = torch.rand((8, 105, 1024), device='cuda')
    y_true = torch.rand((8, 5), device='cuda')

    y_pred = model([x_o1, x_o2, x_w, x_f, x_r])
    loss = criterion(y_pred, y_true)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

