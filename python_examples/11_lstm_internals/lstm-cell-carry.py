import torch
from torch import nn

torch.manual_seed(15)
lstm = nn.LSTM(3, 4, batch_first=True)
first = torch.randn(2, 5, 3, dtype=torch.float32)
second = torch.randn(2, 2, 3, dtype=torch.float32)
_, state = lstm(first)
out, next_state = lstm(second, state)
assert out.shape == (2, 2, 4)
assert next_state[0].shape == next_state[1].shape == (1, 2, 4)
print("second chunk:", tuple(out.shape))
