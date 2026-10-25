import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(10)
tokens = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
lengths = torch.tensor([3, 2], dtype=torch.long)
embedding = nn.Embedding(8, 5, padding_idx=0)
rnn = nn.RNN(5, 7, batch_first=True)
embedded = embedding(tokens)
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
_, hidden = rnn(packed)
assert embedded.shape == (2, 4, 5) and hidden.shape == (1, 2, 7)
print('embedded:', embedded.shape, 'final valid states:', hidden.shape)
