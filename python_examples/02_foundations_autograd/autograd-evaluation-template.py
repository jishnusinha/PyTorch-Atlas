import torch
from torch import nn
model=nn.Linear(3,2); features=torch.ones((4,3))
model.eval()
with torch.inference_mode(): scores=model(features)
assert scores.shape==(4,2) and not scores.requires_grad
snapshot=scores.detach().clone()
assert snapshot.shape==scores.shape
print(tuple(scores.shape),scores.requires_grad)
