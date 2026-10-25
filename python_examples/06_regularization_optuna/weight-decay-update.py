import torch

parameter = torch.nn.Parameter(torch.tensor([2.0]))
optimizer = torch.optim.SGD([parameter], lr=0.1, weight_decay=0.25)
parameter.grad = torch.zeros_like(parameter)
optimizer.step()
assert torch.allclose(parameter, torch.tensor([1.95]))
assert parameter.abs().item() < 2.0
print(f"weight after shrink-only step={parameter.item():.2f}")
