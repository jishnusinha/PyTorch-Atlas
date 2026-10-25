import torch
import torch.nn as nn

class ManualModel:
    def __init__(self):
        self.weight=torch.nn.Parameter(torch.zeros((30,1)))
        self.bias=torch.nn.Parameter(torch.zeros(1))
    def __call__(self,x):
        return x@self.weight+self.bias
class RegisteredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
manual=ManualModel()
registered=RegisteredModel()
with torch.no_grad():
    registered.linear.weight.copy_(manual.weight.T)
    registered.linear.bias.copy_(manual.bias)
x=torch.ones((3,30),dtype=torch.float32)
assert torch.equal(manual(x),registered(x))
assert len(list(registered.parameters()))==2
assert not hasattr(manual,"parameters")
print("same output:",tuple(registered(x).shape),"; registered tensors:",len(list(registered.parameters())))
