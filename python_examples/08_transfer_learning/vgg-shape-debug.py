import torch
from torch import nn

features = torch.zeros(2, 512, 7, 7, dtype=torch.float32)
flat = nn.Flatten()(features)
head = nn.Linear(flat.shape[1], 10)
logits = head(flat)
assert flat.shape == (2, 25088) and logits.shape == (2, 10)
assert head.in_features == 512 * 7 * 7
print(head.in_features, tuple(logits.shape))
