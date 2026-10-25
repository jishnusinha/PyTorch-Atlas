import torch
from torch import nn

torch.manual_seed(9)
cell = nn.RNNCell(2, 3, nonlinearity='tanh')
x_t = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
h_t = torch.zeros(1, 3, dtype=torch.float32)
actual = cell(x_t, h_t)
manual = torch.tanh(x_t @ cell.weight_ih.T + h_t @ cell.weight_hh.T + cell.bias_ih + cell.bias_hh)
assert torch.allclose(actual, manual)
assert actual.shape == (1, 3)
print(actual.detach())
