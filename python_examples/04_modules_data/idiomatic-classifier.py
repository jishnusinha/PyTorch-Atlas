import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(12)
g=torch.Generator().manual_seed(12)
x=torch.randn((128,30),generator=g,dtype=torch.float32)
y=(x[:,0]-0.7*x[:,1]+0.4*x[:,2]>0).float().unsqueeze(1)
class BinaryNet(nn.Module):
    def __init__(self,n_features=30):
        super().__init__()
        self.linear=nn.Linear(n_features,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.15)
with torch.no_grad():
    initial=loss_fn(model(x),y).item()
for _ in range(30):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    loss=loss_fn(logits,y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.inference_mode():
    logits=model(x)
    accuracy=((logits>=0)==y.bool()).float().mean()
assert logits.shape==(128,1) and logits.dtype==torch.float32
assert loss.item()<initial and accuracy.item()>.8
print(f"loss {initial:.3f} -> {loss.item():.3f}; accuracy={accuracy.item():.2f}")
