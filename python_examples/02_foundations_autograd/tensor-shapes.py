import torch
x=torch.arange(6,dtype=torch.float32).reshape(2,3)
row=x[1]
view=x.view(3,2)
batch=view.unsqueeze(0)
assert row.shape==(3,) and batch.shape==(1,3,2)
assert batch.dtype==torch.float32
print(tuple(x.shape),row.tolist(),tuple(batch.shape))
