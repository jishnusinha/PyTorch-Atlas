import torch
from torch import nn

def output_size(n, k, p, s, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

for n, k, p, s, d in [(28, 3, 1, 1, 1), (28, 3, 0, 1, 1), (28, 3, 1, 2, 1), (15, 3, 2, 1, 2)]:
    layer = nn.Conv2d(1, 2, k, stride=s, padding=p, dilation=d)
    y = layer(torch.zeros(1, 1, n, n, dtype=torch.float32))
    assert y.shape[-2:] == (output_size(n, k, p, s, d),) * 2
assert output_size(28, 3, 1, 1) == 28
assert output_size(28, 3, 1, 2) == 14
print("same:", output_size(28, 3, 1, 1), "stride 2:", output_size(28, 3, 1, 2))
