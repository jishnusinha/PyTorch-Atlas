import torch

def make_data(n=240, features=30):
    g=torch.Generator().manual_seed(7)
    x=torch.randn((n,features),generator=g,dtype=torch.float32)
    score=1.5*x[:,0]-1.2*x[:,1]+0.8*x[:,2]
    y=(score>0).to(torch.float32).unsqueeze(1)
    return x,y

x,y=make_data()
g=torch.Generator().manual_seed(11)
train=[]; validation=[]; test=[]
for label in (0.,1.):
    ids=(y[:,0]==label).nonzero().flatten()
    ids=ids[torch.randperm(len(ids),generator=g)]
    a=int(.70*len(ids)); b=int(.85*len(ids))
    train.append(ids[:a]); validation.append(ids[a:b]); test.append(ids[b:])
train_ids=torch.cat(train); val_ids=torch.cat(validation); test_ids=torch.cat(test)
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
assert x_train.shape[1]==30 and y_train.shape==(len(train_ids),1)
assert x_train.dtype==torch.float32 and len(set(train_ids.tolist()).intersection(test_ids.tolist()))==0
print(f"train={tuple(x_train.shape)}, val={tuple(x_val.shape)}, test={tuple(x_test.shape)}")
