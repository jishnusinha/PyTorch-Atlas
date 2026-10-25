import torch
from torch import nn

logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.5, 0.2]], dtype=torch.float32)
labels = torch.tensor([0, 1], dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits, labels)
probabilities = logits.softmax(dim=1)
assert labels.shape == (2,) and labels.dtype == torch.long
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2)) and loss.item() > 0
print('loss:', round(loss.item(), 4), 'probability sums:', probabilities.sum(1))
