import torch
from torch.utils.data import Dataset

class BreastRows(Dataset):
    def __init__(self,features,labels):
        if len(features)!=len(labels):
            raise ValueError("features and labels need the same row count")
        self.features=features.to(torch.float32)
        self.labels=labels.to(torch.float32).reshape(-1,1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
x=torch.arange(360,dtype=torch.float32).reshape(12,30)
y=(torch.arange(12)%2).float()
data=BreastRows(x,y)
row,label=data[3]
assert len(data)==12 and row.shape==(30,) and label.shape==(1,)
assert label.item()==1.0 and row.dtype==torch.float32
print("sample 3:",tuple(row.shape),tuple(label.shape),label.item())
