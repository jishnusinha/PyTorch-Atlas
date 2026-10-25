import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
)
shapes = {}
handles = [layer.register_forward_hook(lambda module, inputs, output, i=i: shapes.update({i: tuple(output.shape)}))
          for i, layer in enumerate(model) if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Flatten, nn.Linear))]
model.eval()
with torch.inference_mode():
    logits = model(torch.zeros(2, 1, 28, 28, dtype=torch.float32))
for handle in handles:
    handle.remove()
assert logits.shape == (2, 10)
assert shapes[0] == (2, 32, 28, 28) and shapes[7] == (2, 64, 7, 7)
assert sum(p.numel() for p in model.parameters()) == 429450
print("traced stages:", shapes, "total:", sum(p.numel() for p in model.parameters()))
