import torch
from torch import nn

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

torch.manual_seed(7)
model = FashionCNN()
model.eval()
with torch.inference_mode():
    logits = model(torch.zeros(2, 1, 28, 28, dtype=torch.float32))
assert logits.shape == (2, 10)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 429450
print("logits:", tuple(logits.shape), "trainable:", 429450)
