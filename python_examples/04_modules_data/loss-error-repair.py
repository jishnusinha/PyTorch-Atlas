import torch
import torch.nn as nn

raw_logits=torch.tensor([0.2,-0.4,1.1],dtype=torch.float32)
raw_labels=torch.tensor([0,1,1],dtype=torch.int64)
logits=raw_logits.reshape(-1,1)
target=raw_labels.to(torch.float32).reshape(-1,1)
loss=nn.BCEWithLogitsLoss()(logits,target)
assert logits.shape==target.shape==(3,1)
assert target.dtype==torch.float32 and torch.isfinite(loss)
print("repaired shapes:",tuple(logits.shape),tuple(target.shape))
