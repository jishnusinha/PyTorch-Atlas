import torch

steps, forget = 20, 0.95
direct = torch.tensor(forget, dtype=torch.float32).pow(steps)
ideal = torch.tensor(1.0, dtype=torch.float32).pow(steps)
assert torch.allclose(direct, torch.tensor(0.3584859), atol=1e-6)
assert ideal.item() == 1.0
print("0.95^20 =", round(direct.item(), 6), "; ideal 1^20 =", ideal.item())
