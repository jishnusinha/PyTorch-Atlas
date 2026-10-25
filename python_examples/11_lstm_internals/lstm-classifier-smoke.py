import torch
from torch import nn

torch.manual_seed(13)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 5, batch_first=True)
        self.head = nn.Linear(5, 3)
    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])
model = Classifier()
x = torch.randn(6, 7, 4, dtype=torch.float32)
y = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
loss = nn.CrossEntropyLoss()(model(x), y)
loss.backward()
assert loss.ndim == 0 and torch.isfinite(loss)
assert all(p.grad is not None for p in model.parameters())
print("loss:", round(loss.item(), 4))
