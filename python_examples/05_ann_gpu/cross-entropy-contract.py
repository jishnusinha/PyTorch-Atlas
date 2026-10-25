import torch
from torch import nn

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.0]])
targets = torch.tensor([0, 1], dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits, targets)
probabilities = logits.softmax(dim=1)
assert logits.shape == (2, 3) and targets.shape == (2,)
assert targets.dtype == torch.long and loss.ndim == 0
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
print(f"loss={loss.item():.4f}, probabilities sum to one per row")
