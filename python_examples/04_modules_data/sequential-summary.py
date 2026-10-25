import importlib.util
import torch
import torch.nn as nn

sequential=nn.Sequential(nn.Linear(30,8),nn.ReLU(),nn.Linear(8,1))
class Explicit(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(30,8)
        self.output=nn.Linear(8,1)
    def forward(self,x):
        return self.output(torch.relu(self.hidden(x)))
model=Explicit()
x=torch.randn((5,30),dtype=torch.float32)
assert sequential(x).shape==model(x).shape==(5,1)
count=sum(p.numel() for p in model.parameters() if p.requires_grad)
assert count==257
if importlib.util.find_spec("torchinfo"):
    from torchinfo import summary
    summary(model,input_size=(5,30))
else:
    print("Optional torchinfo is not installed; parameter count:",count)
print("trainable parameters:",count)
