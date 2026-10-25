import torch
from torch import nn
from torch.utils.data import DataLoader

DOWNLOAD_DATA = False  # Set True only when you intend to fetch Fashion-MNIST.
if not DOWNLOAD_DATA:
    raise SystemExit("Set DOWNLOAD_DATA=True to download the optional dataset.")
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
).to(device)
optimizer, loss_fn = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4), nn.CrossEntropyLoss()
for epoch in range(1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
model.eval()
correct = total = 0
with torch.inference_mode():
    for images, labels in test_loader:
        logits = model(images.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.numel()
assert total == len(test_data)
print(f"one-epoch test accuracy: {correct / total:.3f} ({correct}/{total})")
