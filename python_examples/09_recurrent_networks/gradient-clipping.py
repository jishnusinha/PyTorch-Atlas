import torch
from torch import nn

w = nn.Parameter(torch.tensor([10.0, -10.0], dtype=torch.float32))
loss = (100.0 * w).sum()
loss.backward()
before = w.grad.norm().item()
nn.utils.clip_grad_norm_([w], max_norm=1.0)
after = w.grad.norm().item()
assert before > 100 and abs(after - 1.0) < 1e-5
print(f'before={before:.1f}, after={after:.1f}')
