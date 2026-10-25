import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(14)
x = torch.randn(2, 5, 4, dtype=torch.float32)
lengths = torch.tensor([5, 3], dtype=torch.long)
lstm = nn.LSTM(4, 6, batch_first=True)
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
_, (h, c) = lstm(packed)
assert h.shape == c.shape == (1, 2, 6)
print("valid final state:", tuple(h.shape))
