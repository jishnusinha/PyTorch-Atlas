import torch

c_prev = torch.tensor([2., 3.], dtype=torch.float32, requires_grad=True)
f = torch.tensor([.9, .5], dtype=torch.float32)
i = torch.tensor([.2, .7], dtype=torch.float32)
g = torch.tensor([.4, -.1], dtype=torch.float32)
c = f * c_prev + i * g
c.sum().backward()
assert torch.allclose(c_prev.grad, f)
print("direct-path gradient:", c_prev.grad.tolist())
