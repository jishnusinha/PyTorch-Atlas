import torch
from torch import nn

rnn = nn.RNN(5, 3, batch_first=True)
head = nn.Linear(3, 1)
counts = {name: parameter.numel() for name, parameter in rnn.named_parameters()}
total = sum(counts.values()) + sum(p.numel() for p in head.parameters())
assert counts == {'weight_ih_l0': 15, 'weight_hh_l0': 9, 'bias_ih_l0': 3, 'bias_hh_l0': 3}
assert total == 34
print(counts, 'including head:', total)
