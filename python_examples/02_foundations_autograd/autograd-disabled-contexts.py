import torch
w=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True); x=torch.tensor([[2.,1.]])
tracked=x@w
with torch.no_grad(): plain=x@w
with torch.inference_mode(): inference=x@w
detached=tracked.detach()
assert tracked.requires_grad and not plain.requires_grad
assert not inference.requires_grad and not detached.requires_grad
again=plain.clone().requires_grad_(); again.sum().backward()
assert again.grad is not None
print(True,False,False,False)
