import torch
from torch import nn

vocab = {'<pad>': 0, '<unk>': 1, 'capital': 2}
words = ['capital', 'of', 'france']
ids = [vocab.get(word, vocab['<unk>']) for word in words]
embedding = nn.Embedding(len(vocab), 4, padding_idx=vocab['<pad>'])
vectors = embedding(torch.tensor(ids, dtype=torch.long))
assert ids == [2, 1, 1] and vectors.shape == (3, 4)
assert vocab['<pad>'] != vocab['<unk>']
print(ids, tuple(vectors.shape))
