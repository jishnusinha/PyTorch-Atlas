import torch
x=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
b=torch.tensor([10.,20.,30.])
y=x+b
assert y.shape==(2,3)
assert torch.equal(y[1],torch.tensor([14.,25.,36.]))
print(y.tolist())
