import torch
from torch import nn

lstm = nn.LSTM(4, 3, num_layers=2, bidirectional=True, batch_first=True)
x = torch.randn(2, 5, 4, dtype=torch.float32)
_, (h_n, c_n) = lstm(x)
h_layers = h_n.view(2, 2, 2, 3)
c_layers = c_n.view(2, 2, 2, 3)
assert h_layers.shape == c_layers.shape == (2, 2, 2, 3)
assert torch.allclose(h_n[-2], h_layers[-1, 0])
print("[layers,directions,batch,hidden] =", tuple(h_layers.shape))
