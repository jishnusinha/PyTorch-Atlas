import torch

w=torch.tensor(2.0,dtype=torch.float32,requires_grad=True)
(w**2).backward()
first=w.grad.item()
(3*w).backward()
accumulated=w.grad.item()
assert first==4.0 and accumulated==7.0
w.grad.zero_()
(3*w).backward()
fresh=w.grad.item()
assert fresh==3.0
print(f"iteration 1={first:.0f}; accumulated={accumulated:.0f}; cleared iteration 2={fresh:.0f}")
