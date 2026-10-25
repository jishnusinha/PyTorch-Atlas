import torch
def check(fn,value,slope):
    x=torch.tensor(value,dtype=torch.float32,requires_grad=True)
    y=fn(x); assert y.ndim==0
    y.backward()
    assert torch.allclose(x.grad,torch.tensor(slope))
    return y.item(),x.grad.item()
result=check(lambda x:x**2,3.,6.)
assert result==(9.,6.)
print(result)
