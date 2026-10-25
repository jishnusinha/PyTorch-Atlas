import torch
from torch.utils.data import TensorDataset, DataLoader

def make_loader(x, y, batch_size=4, training=True, drop_last=False):
    assert len(x) == len(y)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=training, drop_last=drop_last,
                      generator=generator, num_workers=0)

x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
y = torch.arange(10, dtype=torch.long)
loader = make_loader(x, y, drop_last=True)
assert [len(a) for a, b in loader] == [4, 4]
print("2 full batches; 2 samples dropped")
