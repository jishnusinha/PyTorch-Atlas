import torch
from torch.utils.data import TensorDataset

x=torch.randn((10,30),generator=torch.Generator().manual_seed(1),dtype=torch.float32)
y=(x[:,0:1]>0).float()
dataset=TensorDataset(x,y)
features,target=dataset[4]
assert len(dataset)==10 and features.shape==(30,) and target.shape==(1,)
assert torch.equal(features,x[4]) and torch.equal(target,y[4])
print("TensorDataset pairs:",tuple(features.shape),tuple(target.shape))
