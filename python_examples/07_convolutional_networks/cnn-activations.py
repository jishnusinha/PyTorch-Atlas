import torch
from torch import nn

x = torch.tensor([[[[-2.0, 3.0]], [[4.0, -5.0]]]], dtype=torch.float32)
relu = nn.ReLU()(x)
bn = nn.BatchNorm2d(2)
bn.train()
y = bn(relu)
assert torch.equal(relu, torch.tensor([[[[0.0, 3.0]], [[4.0, 0.0]]]]))
assert y.shape == x.shape
assert sum(p.numel() for p in bn.parameters()) == 4
assert bn.running_mean.shape == (2,)
print("ReLU:", relu.flatten().tolist(), "BatchNorm affine parameters:", 4)
