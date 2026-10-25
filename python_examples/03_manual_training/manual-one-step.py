import torch
import torch.nn.functional as F

torch.manual_seed(3)
x=torch.randn((8,30),dtype=torch.float32)
y=(x[:,:1]>0).to(torch.float32)
w=torch.zeros((30,1),requires_grad=True); b=torch.zeros((1,),requires_grad=True)
logits=x@w+b; loss=F.binary_cross_entropy_with_logits(logits,y)
before=loss.item(); loss.backward()
assert w.grad.shape==w.shape and b.grad.shape==b.shape
with torch.no_grad():
    w-=0.1*w.grad
    b-=0.1*b.grad
w.grad.zero_(); b.grad.zero_()
after=F.binary_cross_entropy_with_logits(x@w+b,y)
assert after.item()<before and w.grad.count_nonzero()==0
print(f"loss {before:.4f} → {after.item():.4f}; gradients cleared")
