import torch
import torch.nn as nn

torch.manual_seed(2)
x=torch.arange(32,dtype=torch.float32).reshape(16,2)/16
y=(x[:,0:1]+x[:,1:2]>.9).float()
model=nn.Linear(2,1)
optimizer=torch.optim.SGD(model.parameters(),lr=0.3)
loss_fn=nn.BCEWithLogitsLoss()
with torch.no_grad():
    initial=loss_fn(model(x),y).item()
for _ in range(25):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    loss=loss_fn(model(x),y)
    loss.backward()
    assert model.weight.grad is not None
    optimizer.step()
model.eval()
with torch.inference_mode():
    final=loss_fn(model(x),y).item()
assert final<initial
print(f"loss {initial:.3f} -> {final:.3f}")
