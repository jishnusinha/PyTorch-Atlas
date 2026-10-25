import torch

w = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
before = (w - 1.0) ** 2
before.backward()
assert torch.allclose(w.grad, torch.tensor(4.0))
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad.zero_()
after = (w - 1.0) ** 2
assert after.item() < before.item()
assert torch.allclose(w, torch.tensor(2.6))
print(f"weight={w.item():.2f}, loss={after.item():.2f}")
