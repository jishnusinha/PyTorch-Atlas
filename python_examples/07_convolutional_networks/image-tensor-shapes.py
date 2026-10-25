import torch

gray = torch.zeros(8, 28, 28, dtype=torch.float32).unsqueeze(1)
rgb = torch.zeros(8, 32, 32, 3, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
assert gray.shape == (8, 1, 28, 28)
assert rgb.shape == (8, 3, 32, 32)
assert gray.dtype == rgb.dtype == torch.float32
print("NCHW:", tuple(gray.shape), tuple(rgb.shape))
