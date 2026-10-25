import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = nn.Linear(784, 10).to(device)
images = torch.zeros(4, 1, 28, 28).pin_memory() if device.type == "cuda" else torch.zeros(4, 1, 28, 28)
labels = torch.zeros(4, dtype=torch.long)
images = images.to(device, non_blocking=device.type == "cuda").flatten(start_dim=1)
labels = labels.to(device, non_blocking=device.type == "cuda")
logits = model(images)
assert logits.device == labels.device == next(model.parameters()).device
assert logits.shape == (4, 10)
print(f"device={device}, logits={list(logits.shape)}")
