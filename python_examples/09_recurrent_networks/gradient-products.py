import torch

steps = torch.tensor(20.0, dtype=torch.float32)
shrinking = torch.pow(torch.tensor(0.5), steps)
growing = torch.pow(torch.tensor(1.1), steps)
assert shrinking.item() < 1e-6
assert abs(growing.item() - 6.7275) < 0.01
print(f'shrinking={shrinking.item():.8f}, growing={growing.item():.4f}')
