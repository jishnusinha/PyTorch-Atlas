import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

torch.manual_seed(22)
model = nn.Linear(4, 7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
x = torch.randn(3, 4, dtype=torch.float32)
y = torch.tensor([1, 3, 2], dtype=torch.long)
model.train()
optimizer.zero_grad(set_to_none=True)
loss = criterion(model(x), y)
loss.backward()
norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
assert loss.ndim == 0 and torch.isfinite(norm)
print("loss", round(loss.item(), 4))
