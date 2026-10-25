import torch
from torch import nn

x = torch.tensor([[[[1, 3, 2, 0], [9, 2, 6, 8], [4, 1, 3, 2], [0, 5, 7, 1]]]], dtype=torch.float32)
maximum = nn.MaxPool2d(2, 2)(x)
average = nn.AvgPool2d(2, 2)(x)
global_average = nn.AdaptiveAvgPool2d((1, 1))(x)
assert maximum.flatten().tolist() == [9, 8, 5, 7]
assert torch.allclose(average.flatten(), torch.tensor([3.75, 4.0, 2.5, 3.25]))
assert global_average.shape == (1, 1, 1, 1)
assert torch.isclose(global_average.squeeze(), torch.tensor(3.375))
print("max:", maximum.flatten().tolist(), "average:", average.flatten().tolist(), "global:", global_average.item())
