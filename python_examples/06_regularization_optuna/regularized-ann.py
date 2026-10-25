import torch
from torch import nn

torch.manual_seed(5)
model = nn.Sequential(
    nn.Linear(20, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
images, labels = torch.randn(16, 20), torch.arange(16) % 10
model.train()
optimizer.zero_grad(set_to_none=True)
loss = nn.CrossEntropyLoss()(model(images), labels)
loss.backward()
optimizer.step()
model.eval()
with torch.inference_mode():
    logits = model(images)
assert logits.shape == (16, 10) and loss.isfinite()
print(f"loss={loss.item():.3f}, eval logits={list(logits.shape)}")
