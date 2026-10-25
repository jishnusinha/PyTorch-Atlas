import torch
def add_offset(values,offset):
    assert values.ndim==2 and offset.ndim==1
    assert values.shape[-1]==offset.shape[0]
    return values+offset
y=add_offset(torch.zeros((5,4)),torch.arange(4,dtype=torch.float32))
assert y.shape==(5,4)
print(tuple(y.shape))
