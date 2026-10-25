import torch
from torch import nn

torch.manual_seed(17)
rnn = nn.RNN(2, 4, batch_first=True)
head = nn.Linear(4, 1)
x = torch.randn(2, 3, 2, dtype=torch.float32)
target = torch.tensor([[1.], [0.]], dtype=torch.float32)
sequence, _ = rnn(x)
logits = head(sequence[:, -1])
loss = nn.BCEWithLogitsLoss()(logits, target)
loss.backward()
grad = rnn.weight_hh_l0.grad
assert grad is not None and torch.isfinite(grad).all()
assert grad.norm().item() > 0
print('loss:', round(loss.item(), 4), 'recurrent grad norm:', round(grad.norm().item(), 4))
