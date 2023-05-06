import torch
torch.cuda.set_device(0)

from linear_mult.models.MulT import MulT


model = MulT(input_modality_channels=[35, 25, 768], 
             output_dim=5, 
             attention_type='linear', add_cls_token=False, target_sequence=False).cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters())

for i in range(5):
    print(f'epoch: {i}')
    x_a = torch.rand((32, 1500, 35), device='cuda')
    x_v = torch.rand((32, 450, 25), device='cuda')
    x_t = torch.rand((32, 105, 768), device='cuda')
    y_true = torch.rand((32, 5), device='cuda')

    y_pred = model([x_a, x_v, x_t])
    loss = criterion(y_pred, y_true)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

