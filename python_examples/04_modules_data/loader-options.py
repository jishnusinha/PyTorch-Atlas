import torch
from torch.utils.data import DataLoader,RandomSampler,TensorDataset

x=torch.arange(20,dtype=torch.float32).reshape(10,2)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
train=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0,
                 drop_last=False,pin_memory=torch.cuda.is_available())
random_sampler=RandomSampler(dataset,generator=torch.Generator().manual_seed(8))
sampled=DataLoader(dataset,batch_size=4,sampler=random_sampler,num_workers=0)
evaluation=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
dropped=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=True,num_workers=0)
assert [len(by) for _,by in train]==[4,4,2]
assert [len(by) for _,by in evaluation]==[4,4,2]
assert [len(by) for _,by in dropped]==[4,4]
assert next(iter(evaluation))[1][0].item()==0.0
sampled_ids=torch.cat([by.flatten() for _,by in sampled]).long().tolist()
assert sorted(sampled_ids)==list(range(10))
print("train/eval keep 10 rows; drop_last keeps",sum(len(by) for _,by in dropped))
