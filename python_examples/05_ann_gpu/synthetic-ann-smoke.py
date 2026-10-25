import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(17)
device = torch.device("cpu")
labels = torch.arange(10).repeat_interleave(20)
images = torch.randn(200, 1, 28, 28) * 0.15
for label in range(10):
    images[labels == label, 0, 0, label] += 2.5
train_x, train_y = images[:150], labels[:150]
valid_x, valid_y = images[150:], labels[150:]
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=30, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=25)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
first_loss = loss_fn(model(train_x.to(device)), train_y.to(device)).item()
for _ in range(8):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
correct = total = 0
confusion = torch.zeros(10, 10, dtype=torch.long)
with torch.inference_mode():
    for batch_x, batch_y in valid_loader:
        logits = model(batch_x.to(device))
        predictions = logits.argmax(dim=1).cpu()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.numel()
        confusion.index_put_((batch_y, predictions), torch.ones_like(batch_y), accumulate=True)
final_loss = loss_fn(model(train_x.to(device)), train_y.to(device)).item()
assert final_loss < first_loss and total == 50
assert confusion.sum().item() == total
print(f"train loss {first_loss:.3f} → {final_loss:.3f}; validation accuracy={correct / total:.2%}")
