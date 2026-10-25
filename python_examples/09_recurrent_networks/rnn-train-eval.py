import torch
from torch import nn

torch.manual_seed(21)
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 4, batch_first=True)
        self.head = nn.Linear(4, 1)
    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.head(h_n[-1])
model = Classifier()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
x = torch.randn(8, 5, 3, dtype=torch.float32)
y = torch.randint(0, 2, (8, 1)).float()
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = nn.BCEWithLogitsLoss()(logits, y)
loss.backward()
optimizer.step()
model.eval()
with torch.inference_mode():
    prediction = model(x)
assert prediction.shape == (8, 1) and torch.isfinite(prediction).all()
print(tuple(prediction.shape))
