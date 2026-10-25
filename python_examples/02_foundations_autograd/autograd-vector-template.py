import torch
x=torch.tensor([1.,2.,3.],requires_grad=True)
y=2*x
upstream=torch.tensor([1.,.5,-1.])
y.backward(gradient=upstream)
assert torch.equal(x.grad,torch.tensor([2.,1.,-2.]))
print(x.grad.tolist())
