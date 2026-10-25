import torch
w=torch.tensor(2.,requires_grad=True)
(w**2).backward(); assert w.grad.item()==4
(3*w).backward(); assert w.grad.item()==7
w.grad=None
(3*w).backward()
assert w.grad.item()==3
print("accumulated=7; cleared=3")
