import torch

raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
pixels = raw.float() / 255.0
images = torch.zeros(4, 1, 28, 28)
flat = images.flatten(start_dim=1)
labels = torch.tensor([0, 1, 8, 9], dtype=torch.long)
assert pixels.min() >= 0 and pixels.max() <= 1
assert flat.shape == (4, 784) and labels.shape == (4,)
assert labels.dtype == torch.long
print(f"pixels={pixels.tolist()}, flat={list(flat.shape)}, labels={labels.dtype}")
