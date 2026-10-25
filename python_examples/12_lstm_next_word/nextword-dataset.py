import torch
from torch.utils.data import Dataset, DataLoader

class PrefixDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, index): return self.x[index], self.y[index]

x = torch.tensor([[0, 4, 8], [0, 5, 6]], dtype=torch.long)
y = torch.tensor([8, 6], dtype=torch.long)
loader = DataLoader(PrefixDataset(x, y), batch_size=2, shuffle=False)
batch_x, batch_y = next(iter(loader))
assert batch_x.shape == (2, 3) and batch_y.shape == (2,)
assert batch_x.dtype == batch_y.dtype == torch.long
print(tuple(batch_x.shape), tuple(batch_y.shape))
