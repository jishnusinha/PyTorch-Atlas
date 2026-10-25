import torch

def fit_scalar(initial=3.0, target=1.0, lr=0.1, steps=20):
    w = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
    for _ in range(steps):
        loss = (w - target) ** 2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad = None
    return w.detach()

result = fit_scalar()
assert abs(result.item() - 1.0) < 0.03
print(f"weight={result.item():.3f}")
