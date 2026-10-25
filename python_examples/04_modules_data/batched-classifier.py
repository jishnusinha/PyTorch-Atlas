import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
torch.manual_seed(22)
g=torch.Generator().manual_seed(22)
x=torch.randn((180,30),generator=g,dtype=torch.float32)
y=(x[:,0]-0.8*x[:,1]+0.5*x[:,2]>0).float().reshape(-1,1)
ids=torch.randperm(len(x),generator=g)
train_ids,val_ids,test_ids=ids[:120],ids[120:150],ids[150:]
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
train_loader=DataLoader(TensorDataset(x_train,y_train),batch_size=16,shuffle=True,generator=g)
val_loader=DataLoader(TensorDataset(x_val,y_val),batch_size=16,shuffle=False)
test_loader=DataLoader(TensorDataset(x_test,y_test),batch_size=16,shuffle=False)
class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.2)
def evaluate(loader):
    model.eval()
    total=correct=0
    with torch.inference_mode():
        for features,target in loader:
            logits=model(features)
            correct+=((logits>=0)==target.bool()).sum().item()
            total+=len(target)
    return correct/total
with torch.no_grad():
    initial=loss_fn(model(x_train),y_train).item()
for _ in range(12):
    model.train()
    for features,target in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(features),target)
        loss.backward()
        optimizer.step()
model.eval()
with torch.inference_mode():
    final=loss_fn(model(x_train),y_train).item()
val_accuracy=evaluate(val_loader)
test_accuracy=evaluate(test_loader)
assert x_train.shape[1]==30 and y_train.shape[1]==1
assert x_train.dtype==torch.float32 and final<initial
assert 0.0<=val_accuracy<=1.0 and 0.0<=test_accuracy<=1.0
assert val_accuracy>.75 and test_accuracy>.75
print(f"train loss {initial:.3f} -> {final:.3f}; val={val_accuracy:.2f}; test={test_accuracy:.2f}")
