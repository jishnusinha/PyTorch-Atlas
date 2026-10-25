import torch
from torch import nn

layer = nn.Conv2d(1, 2, kernel_size=3, bias=True)
x = torch.randn(2, 1, 5, 5, dtype=torch.float32)
y = layer(x)
assert layer.weight.shape == (2, 1, 3, 3)
assert y.shape == (2, 2, 3, 3)
assert sum(p.numel() for p in layer.parameters()) == 20
print("weights:", tuple(layer.weight.shape), "output:", tuple(y.shape), "parameters:", 20)
