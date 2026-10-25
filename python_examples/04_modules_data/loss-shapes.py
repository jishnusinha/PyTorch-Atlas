import torch
import torch.nn as nn

logits=torch.tensor([[-2.0],[0.0],[2.0]],dtype=torch.float32)
target=torch.tensor([[0.0],[1.0],[1.0]],dtype=torch.float32)
stable=nn.BCEWithLogitsLoss()(logits,target)
probability=logits.sigmoid()
classic=nn.BCELoss()(probability,target)
assert torch.allclose(stable,classic,atol=1e-6)
shape_error=False
try:
    nn.BCEWithLogitsLoss()(logits,target.squeeze(1))
except (ValueError,RuntimeError):
    shape_error=True
assert shape_error  # [B,1] logits do not pair with [B] labels
type_error=False
try:
    nn.BCEWithLogitsLoss()(logits,target.long())
except (RuntimeError,TypeError):
    type_error=True
assert type_error  # binary BCE targets should be floating point
print(f"both valid losses={stable.item():.4f}; mismatches caught")
