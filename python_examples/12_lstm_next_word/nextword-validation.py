import torch
from torch import nn

model = nn.Linear(3, 5)
criterion = nn.CrossEntropyLoss()
x = torch.randn(4, 3, dtype=torch.float32)
y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
model.eval()
with torch.inference_mode():
    loss = criterion(model(x), y)
assert torch.isfinite(loss) and all(p.grad is None for p in model.parameters())
print("validation loss", round(loss.item(), 4))
