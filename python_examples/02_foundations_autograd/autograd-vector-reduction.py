import torch
x=torch.tensor([1.,2.,3.],requires_grad=True)
t=torch.tensor([0.,1.,1.])
r=x-t; loss=(r**2).sum(); loss.backward()
assert torch.equal(x.grad,torch.tensor([2.,2.,4.]))
print(r.tolist(),loss.item(),x.grad.tolist())
