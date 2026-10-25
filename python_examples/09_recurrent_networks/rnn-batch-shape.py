import torch
from torch import nn

torch.manual_seed(42)
rnn = nn.RNN(input_size=3, hidden_size=5, batch_first=True)
x = torch.randn(2, 4, 3, dtype=torch.float32)
output, h_n = rnn(x)
assert output.shape == (2, 4, 5)
assert h_n.shape == (1, 2, 5)
assert torch.allclose(output[:, -1, :], h_n[0])
print(tuple(output.shape), tuple(h_n.shape))
