import torch
from torch import nn

torch.manual_seed(7)
class SentimentToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(2, 5, batch_first=True)
        self.head = nn.Linear(5, 1)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.head(h_n[-1])
model = SentimentToy()
x = torch.randn(6, 4, 2, dtype=torch.float32)
y = torch.tensor([[0.], [1.], [0.], [1.], [1.], [0.]], dtype=torch.float32)
logits = model(x)
loss = nn.BCEWithLogitsLoss()(logits, y)
assert logits.shape == y.shape == (6, 1) and torch.isfinite(loss)
print(tuple(logits.shape), loss.item())
