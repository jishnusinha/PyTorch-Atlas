import torch

c_prev = torch.tensor([4., 5., 6.], dtype=torch.float32)
f = torch.tensor([0., 1., .5], dtype=torch.float32)
i = torch.tensor([1., 0., 0.], dtype=torch.float32)
g = torch.tensor([.8, -.4, .2], dtype=torch.float32)
o = torch.tensor([1., 0., .5], dtype=torch.float32)
c = f * c_prev + i * g
h = o * torch.tanh(c)
assert torch.allclose(c, torch.tensor([.8, 5., 3.]))
assert torch.allclose(h, torch.tensor([.664, 0., .498]), atol=.002)
print("cell:", c.tolist(), "hidden:", h.tolist())
