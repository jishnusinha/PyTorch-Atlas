import torch

x = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
y = x ** 2
y.backward()
assert x.grad is not None
assert torch.allclose(x.grad, torch.tensor(6.0))
print(f"value={y.item():.1f}, gradient={x.grad.item():.1f}")
