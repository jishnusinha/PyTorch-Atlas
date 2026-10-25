import torch

def check_derivative(fn, analytic, value):
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    fn(x).backward()
    expected = analytic(x.detach())
    assert torch.allclose(x.grad, expected, atol=1e-5)
    return x.grad.item()

print(check_derivative(lambda x: x ** 2, lambda x: 2 * x, 3.0))
