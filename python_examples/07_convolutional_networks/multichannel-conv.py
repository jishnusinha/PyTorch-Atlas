import torch
from torch import nn

layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)
x = torch.randn(4, 3, 32, 32, dtype=torch.float32)
y = layer(x)
expected_parameters = ((3 * 3 * 3) + 1) * 16
assert tuple(layer.weight.shape) == (16, 3, 3, 3)
assert y.shape == (4, 16, 32, 32)
assert sum(p.numel() for p in layer.parameters()) == expected_parameters == 448
print("output:", tuple(y.shape), "parameters:", expected_parameters)
