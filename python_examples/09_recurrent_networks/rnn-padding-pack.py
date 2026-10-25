import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(3)
tokens = torch.tensor([[1, 2, 3, 0, 0], [2, 4, 1, 3, 0], [5, 2, 0, 0, 0]])
lengths = torch.tensor([3, 4, 2])
embedding = nn.Embedding(6, 4, padding_idx=0)
rnn = nn.RNN(4, 6, batch_first=True)
embedded = embedding(tokens)
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
_, h_n = rnn(packed)
assert embedded.shape == (3, 5, 4) and h_n.shape == (1, 3, 6)
print(tuple(embedded.shape), tuple(h_n.shape))
