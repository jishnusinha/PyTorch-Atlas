import torch
from torch.utils.data import DataLoader,SequentialSampler,TensorDataset

x=torch.arange(300,dtype=torch.float32).reshape(10,30)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
sampler=SequentialSampler(dataset)
loader=DataLoader(dataset,batch_size=4,sampler=sampler,drop_last=False)
batches=list(loader)
assert [len(batch_y) for _,batch_y in batches]==[4,4,2]
assert batches[0][1].flatten().tolist()==[0.,1.,2.,3.]
assert batches[-1][0].shape==(2,30)
print("batch sizes:",[len(labels) for _,labels in batches])
