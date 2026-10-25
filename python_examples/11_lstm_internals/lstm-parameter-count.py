import torch
from torch import nn

input_size, hidden, layers, directions = 4, 3, 2, 2
lstm = nn.LSTM(input_size, hidden, num_layers=layers, bidirectional=True, batch_first=True)
total = sum(parameter.numel() for parameter in lstm.parameters())
expected = 2 * (4 * hidden * (input_size + hidden + 2) + 4 * hidden * (2 * hidden + hidden + 2))
assert total == expected == 480
print("parameters:", total)
