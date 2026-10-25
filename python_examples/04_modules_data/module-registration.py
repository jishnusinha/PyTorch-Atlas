import torch
import torch.nn as nn

class BinaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
        self.temperature=nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        return self.linear(x)/self.temperature.clamp_min(0.1)
model=BinaryHead()
seen=[]
handle=model.linear.register_forward_hook(lambda module,args,out: seen.append(tuple(out.shape)))
x=torch.zeros((4,30),dtype=torch.float32)
logits=model(x)  # Module.__call__ invokes forward and its hooks
handle.remove()
keys=tuple(model.state_dict().keys())
names=tuple(name for name,_ in model.named_parameters())
assert logits.shape==(4,1) and seen==[(4,1)]
assert set(keys)=={"temperature","linear.weight","linear.bias"}
assert set(names)==set(keys)
assert sum(p.numel() for p in model.parameters())==32
print("registered:",", ".join(keys))
