import torch

image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
flat = image.flatten(start_dim=1)
patch = image[:, :, 1:3, 1:3]
ann_parameters = (4 * 4 + 1) * 8
shared_filter_parameters = 3 * 3 + 1
assert flat.shape == (1, 16) and patch.shape == (1, 1, 2, 2)
assert ann_parameters == 136 and shared_filter_parameters == 10
print("flat:", flat.shape, "patch:", patch.shape)
print("dense weights:", ann_parameters, "one shared filter:", shared_filter_parameters)
