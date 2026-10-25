import torch
from torch import nn

torch.manual_seed(13)
class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(2, 6, batch_first=True)
        self.head = nn.Linear(6, 5)
    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(sequence)
model = Tagger()
x = torch.randn(3, 4, 2, dtype=torch.float32)
labels = torch.randint(0, 5, (3, 4), dtype=torch.long)
logits = model(x)
loss = nn.CrossEntropyLoss()(logits.reshape(-1, 5), labels.reshape(-1))
assert logits.shape == (3, 4, 5) and torch.isfinite(loss)
print(tuple(logits.shape), labels.shape)
