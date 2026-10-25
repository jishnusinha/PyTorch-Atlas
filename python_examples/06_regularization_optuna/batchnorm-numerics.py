import torch
from torch import nn

norm = nn.BatchNorm1d(2, affine=True, track_running_stats=True)
norm.eval()
with torch.no_grad():
    norm.running_mean.copy_(torch.tensor([1.0, 2.0]))
    norm.running_var.copy_(torch.tensor([4.0, 9.0]))
    norm.weight.copy_(torch.tensor([2.0, 0.5]))
    norm.bias.copy_(torch.tensor([1.0, -1.0]))
x = torch.tensor([[3.0, 5.0]])
y = norm(x)
expected = torch.tensor([[2.9999975, -0.4999997]])
assert torch.allclose(y, expected, atol=1e-5)
print(f"normalized and affine output={y.tolist()}")
