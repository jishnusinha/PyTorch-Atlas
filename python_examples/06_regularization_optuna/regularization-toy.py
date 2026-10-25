import torch
from torch import nn

torch.manual_seed(31)
block = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(8, 4))
x = torch.randn(16, 8)
block.train()
train_a, train_b = block(x), block(x)
assert train_a.shape == (16, 4) and not torch.equal(train_a, train_b)
block.eval()
eval_a, eval_b = block(x), block(x)
assert torch.equal(eval_a, eval_b)
assert block[1].num_batches_tracked.item() > 0
print(f"output={list(eval_a.shape)}; dropout varies in train and is disabled in eval")
