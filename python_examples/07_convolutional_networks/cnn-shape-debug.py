import torch
from torch import nn

features = torch.zeros(4, 64, 7, 7, dtype=torch.float32)
flat = nn.Flatten()(features)
head = nn.Linear(flat.shape[1], 10)
logits = head(flat)
assert flat.shape == (4, 3136) and logits.shape == (4, 10)
assert head.in_features == 64 * 7 * 7
print("measured in_features:", flat.shape[1], "logits:", tuple(logits.shape))
