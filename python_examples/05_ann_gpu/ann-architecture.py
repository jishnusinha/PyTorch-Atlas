import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)
logits = model(torch.zeros(5, 1, 28, 28))
parameters = sum(p.numel() for p in model.parameters())
expected = (784 + 1) * 128 + (128 + 1) * 64 + (64 + 1) * 10
assert logits.shape == (5, 10)
assert parameters == expected == 109386
print(f"logits={list(logits.shape)}, parameters={parameters}")
