import torch
from torch import nn

torch.manual_seed(19)
torch.set_num_threads(1)
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10),
)
x = torch.rand(8, 1, 28, 28, dtype=torch.float32)
y = torch.arange(8, dtype=torch.long) % 10
optimizer, loss_fn = torch.optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
model.train()
for start in range(0, 8, 4):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x[start:start + 4])
    loss = loss_fn(logits, y[start:start + 4])
    loss.backward()
    optimizer.step()
model.eval()
with torch.inference_mode():
    scores = model(x)
assert scores.shape == (8, 10) and torch.isfinite(scores).all()
print("one synthetic epoch complete:", tuple(scores.shape))
