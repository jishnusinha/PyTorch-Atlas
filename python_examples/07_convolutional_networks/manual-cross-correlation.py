import torch
import torch.nn.functional as F

x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
weight = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
manual = torch.empty(2, 2, dtype=torch.float32)
for row in range(2):
    for col in range(2):
        patch = x[0, 0, row:row + 3, col:col + 3]
        manual[row, col] = (patch * weight[0, 0]).sum()
library = F.conv2d(x, weight)
assert torch.equal(manual, library[0, 0])
assert torch.equal(manual, torch.full((2, 2), -6.0))
print("manual output:", manual.tolist())
