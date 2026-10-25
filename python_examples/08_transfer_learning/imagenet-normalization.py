import torch
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
pixel = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32).view(3, 1, 1)
normalized = transforms.Normalize(mean, std)(pixel)
expected = torch.tensor([(0.8 - m) / s for m, s in zip(mean, std)])
assert torch.allclose(normalized[:, 0, 0], expected)
assert abs(normalized[0, 0, 0].item() - 1.3755) < 0.002
print(normalized[:, 0, 0].tolist())
