import torch
def prepare(x,features):
    assert x.ndim==2 and x.shape[1]==features
    return x.to(dtype=torch.float32,device="cpu").contiguous()
x=prepare(torch.ones((4,3),dtype=torch.float64),3)
assert x.shape==(4,3) and x.dtype==torch.float32
print(x.shape,x.dtype)
