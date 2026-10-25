import torch
from torch import nn

def lstm_parameter_count(input_size, hidden, layers=1, directions=1):
    total = 0
    for layer in range(layers):
        width = input_size if layer == 0 else directions * hidden
        total += directions * 4 * hidden * (width + hidden + 2)
    return total
lstm = nn.LSTM(5, 4, num_layers=3, bidirectional=True)
count = sum(p.numel() for p in lstm.parameters())
assert count == lstm_parameter_count(5, 4, 3, 2)
print("formula and module:", count)
