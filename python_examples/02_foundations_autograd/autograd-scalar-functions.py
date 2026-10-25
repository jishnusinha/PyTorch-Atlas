import math
import torch
x=torch.tensor(.7,dtype=torch.float32,requires_grad=True)
y=x**2; z=torch.sin(y); z.backward()
expected=2*.7*math.cos(.49)
assert torch.allclose(x.grad,torch.tensor(expected),atol=1e-6)
assert not y.is_leaf
print(y.item(),z.item(),x.grad.item())
