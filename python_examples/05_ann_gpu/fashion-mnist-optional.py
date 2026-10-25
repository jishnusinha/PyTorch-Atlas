"""Optional real-data exercise. Set DOWNLOAD_DATA=True to fetch missing files."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DOWNLOAD_DATA = False  # Change to True only when you want torchvision to download the dataset.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()  # Converts uint8 pixels to float in [0, 1].
train_data = datasets.FashionMNIST("data", train=True, download=DOWNLOAD_DATA, transform=transform)
test_data = datasets.FashionMNIST("data", train=False, download=DOWNLOAD_DATA, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=device.type == "cuda")
test_loader = DataLoader(test_data, batch_size=128, pin_memory=device.type == "cuda")
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for images, labels in train_loader:
    images = images.to(device, non_blocking=device.type == "cuda")
    labels = labels.to(device, non_blocking=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(images), labels)  # Raw logits; no softmax before cross entropy.
    loss.backward()
    optimizer.step()
model.eval()
correct = total = 0
assert len(train_loader) > 0 and len(test_loader) > 0
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
print(f"test accuracy={correct / total:.2%} over {total} images")
