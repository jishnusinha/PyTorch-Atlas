import torch
from torch import nn

torch.manual_seed(8)
features = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6), nn.ReLU())
for parameter in features.parameters():
    parameter.requires_grad = False
model = nn.Sequential(features, nn.Linear(6, 3))
x = torch.rand(4, 3, 8, 8, dtype=torch.float32)
labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
assert logits.shape == (4, 3) and torch.isfinite(loss)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 21
print(tuple(logits.shape), loss.item())
