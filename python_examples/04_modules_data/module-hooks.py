import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(30,8)
        self.output=nn.Linear(8,1)
    def forward(self,x):
        return self.output(torch.relu(self.hidden(x)))
model=MLP()
observed=[]
hook=model.hidden.register_forward_hook(lambda m,inputs,out: observed.append(out.shape))
prediction=model(torch.ones((2,30)))
hook.remove()
assert observed==[torch.Size([2,8])]
assert prediction.shape==(2,1)
print("hook saw",tuple(observed[0]),"; prediction",tuple(prediction.shape))
