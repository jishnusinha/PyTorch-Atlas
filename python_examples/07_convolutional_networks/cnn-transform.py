import torch

pixels = torch.arange(2 * 784, dtype=torch.float32).reshape(2, 784).remainder(256)
images = (pixels / 255.0).reshape(-1, 1, 28, 28)
labels = torch.tensor([3, 7], dtype=torch.long)
assert images.shape == (2, 1, 28, 28)
assert images.dtype == torch.float32 and labels.dtype == torch.long
assert images.min() >= 0 and images.max() <= 1
print("images:", tuple(images.shape), images.dtype, "labels:", labels.tolist())
