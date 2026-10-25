import torch
from torch import nn
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split

ALLOW_DOWNLOADS = False
if not ALLOW_DOWNLOADS:
    raise SystemExit('Set ALLOW_DOWNLOADS=True to fetch weights and Fashion-MNIST.')
torch.manual_seed(42)
weights = models.VGG16_Weights.DEFAULT
transform = weights.transforms()
dataset_transform = lambda image: transform(image.convert('RGB'))
full = datasets.FashionMNIST('./data', train=True, download=True, transform=dataset_transform)
train_set, val_set = random_split(full, [54000, 6000], generator=torch.Generator().manual_seed(42))
test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=dataset_transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(weights=weights)
for parameter in model.features.parameters():
    parameter.requires_grad = False
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
model = model.to(device)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
best_loss, best_state = float('inf'), None
for epoch in range(1):
    model.train()
    model.features.eval()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in val_loader:
            total_loss += loss_fn(model(images.to(device)), labels.to(device)).item() * labels.size(0)
    val_loss = total_loss / len(val_set)
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
model.eval()
correct = total = 0
with torch.inference_mode():
    for images, labels in test_loader:
        correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
        total += labels.numel()
assert total == len(test_set)
print('held-out accuracy:', correct / total)
