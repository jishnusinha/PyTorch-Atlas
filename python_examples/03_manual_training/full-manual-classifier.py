import torch

def dataset(n=240):
    g=torch.Generator().manual_seed(7)
    x=torch.randn((n,30),generator=g,dtype=torch.float32)
    score=1.5*x[:,0]-1.2*x[:,1]+0.8*x[:,2]
    return x,(score>0).to(torch.float32).unsqueeze(1)
def bce(prob,y):
    p=prob.clamp(1e-7,1-1e-7)
    return -(y*torch.log(p)+(1-y)*torch.log(1-p)).mean()
class MySimpleNN:
    def __init__(self):
        self.weights=(torch.randn((30,1))*0.05).requires_grad_()
        self.bias=torch.zeros((1,),requires_grad=True)
    def forward(self,x):
        logits=x@self.weights+self.bias
        return logits,torch.sigmoid(logits)
    def __call__(self,x):
        return self.forward(x)

torch.manual_seed(13)
x,y=dataset()
g=torch.Generator().manual_seed(11)
sets=[[],[],[]]
for label in (0.,1.):
    ids=(y[:,0]==label).nonzero().flatten()
    ids=ids[torch.randperm(len(ids),generator=g)]
    a=int(.70*len(ids)); b=int(.85*len(ids))
    for bucket,part in zip(sets,(ids[:a],ids[a:b],ids[b:])):
        bucket.append(part)
train_ids,val_ids,test_ids=[torch.cat(parts) for parts in sets]
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
model=MySimpleNN()
initial=bce(model(x_train)[1],y_train).item()
learning_rate=0.12
for epoch in range(100):
    logits,probabilities=model(x_train)
    loss=bce(probabilities,y_train)
    loss.backward()
    with torch.no_grad():
        model.weights-=learning_rate*model.weights.grad
        model.bias-=learning_rate*model.bias.grad
    model.weights.grad.zero_(); model.bias.grad.zero_()
final=bce(model(x_train)[1],y_train)
with torch.inference_mode():
    test_logits,test_probabilities=model(x_test)
    predicted=(test_probabilities>=0.5).to(torch.float32)
    accuracy=(predicted==y_test).to(torch.float32).mean()
assert model.weights.shape==(30,1) and model.bias.shape==(1,)
assert x_train.dtype==y_train.dtype==torch.float32
assert test_logits.shape==test_probabilities.shape==y_test.shape
assert final.item()<initial and accuracy.item()>.75
print(f"train loss {initial:.4f} → {final.item():.4f}; test accuracy={accuracy.item():.3f}; test shape={tuple(predicted.shape)}")
