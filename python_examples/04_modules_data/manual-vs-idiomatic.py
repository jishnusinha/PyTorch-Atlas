import torch
import torch.nn as nn

torch.manual_seed(4)
x=torch.tensor([[1.0],[2.0]],dtype=torch.float32)
y=torch.tensor([[0.0],[1.0]],dtype=torch.float32)
w=torch.tensor([[0.2]],requires_grad=True)
b=torch.tensor([0.1],requires_grad=True)
layer=nn.Linear(1,1)
with torch.no_grad():
    layer.weight.copy_(w)
    layer.bias.copy_(b)
manual_loss=nn.functional.mse_loss(x@w+b,y)
manual_loss.backward()
manual_w=w.detach()-0.05*w.grad
model_loss=nn.functional.mse_loss(layer(x),y)
opt=torch.optim.SGD(layer.parameters(),lr=0.05)
opt.zero_grad(set_to_none=True)
model_loss.backward()
opt.step()
assert torch.allclose(manual_w,layer.weight)
assert torch.allclose(w.grad,layer.weight.grad)
print(f"manual={manual_w.item():.4f}; SGD={layer.weight.item():.4f}")
