import torch
a=torch.tensor(2.,requires_grad=True); b=torch.tensor(3.,requires_grad=True)
p=a*b; p.retain_grad(); loss=p**2; loss.backward()
assert a.grad.item()==36 and b.grad.item()==24
assert p.grad.item()==12
print(a.grad.item(),b.grad.item(),p.grad.item())
