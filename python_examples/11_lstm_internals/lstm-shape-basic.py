import torch
from torch import nn

torch.manual_seed(12)
layer = nn.LSTM(input_size=4, hidden_size=6, num_layers=2, bidirectional=True, batch_first=True)
x = torch.randn(3, 5, 4, dtype=torch.float32)
y, (h_n, c_n) = layer(x)
assert y.shape == (3, 5, 12)
assert h_n.shape == c_n.shape == (4, 3, 6)
print("output:", tuple(y.shape), "states:", tuple(h_n.shape))
