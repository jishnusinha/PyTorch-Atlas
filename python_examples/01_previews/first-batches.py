import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
y = torch.arange(10, dtype=torch.long)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
sizes = [features.shape[0] for features, labels in loader]
assert sizes == [4, 4, 2]
assert len(loader) == 3
print(sizes)
