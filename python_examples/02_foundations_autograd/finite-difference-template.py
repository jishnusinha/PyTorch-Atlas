import torch
def compare(fn,value,h=1e-3):
    x=torch.tensor(value,dtype=torch.float32,requires_grad=True)
    fn(x).backward()
    with torch.no_grad():
        numeric=(fn(torch.tensor(value+h)).item()-fn(torch.tensor(value-h)).item())/(2*h)
    assert abs(x.grad.item()-numeric)<2e-3
    return x.grad.item(),numeric
print(compare(torch.sin,.4))
