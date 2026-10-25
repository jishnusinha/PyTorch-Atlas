import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(11)
cell = nn.LSTMCell(4, 3)
x = torch.randn(2, 4, dtype=torch.float32)
h = torch.randn(2, 3, dtype=torch.float32)
c = torch.randn(2, 3, dtype=torch.float32)
packed = F.linear(x, cell.weight_ih, cell.bias_ih) + F.linear(h, cell.weight_hh, cell.bias_hh)
i_pre, f_pre, g_pre, o_pre = packed.chunk(4, dim=1)
c_manual = torch.sigmoid(f_pre) * c + torch.sigmoid(i_pre) * torch.tanh(g_pre)
h_manual = torch.sigmoid(o_pre) * torch.tanh(c_manual)
h_native, c_native = cell(x, (h, c))
assert torch.allclose(c_manual, c_native, atol=1e-6)
assert torch.allclose(h_manual, h_native, atol=1e-6)
print("manual and nn.LSTMCell agree")
