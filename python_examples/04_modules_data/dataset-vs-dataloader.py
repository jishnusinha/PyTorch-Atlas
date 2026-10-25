import torch
from torch.utils.data import DataLoader,TensorDataset

x=torch.arange(300,dtype=torch.float32).reshape(10,30)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
one_x,one_y=dataset[3]
batch_x,batch_y=next(iter(DataLoader(dataset,batch_size=4,shuffle=False)))
assert one_x.shape==(30,) and one_y.shape==(1,)
assert batch_x.shape==(4,30) and batch_y.shape==(4,1)
assert torch.equal(batch_x[3],one_x) and batch_y[3].item()==one_y.item()
print("one sample",tuple(one_x.shape),tuple(one_y.shape),"; batch",tuple(batch_x.shape),tuple(batch_y.shape))
