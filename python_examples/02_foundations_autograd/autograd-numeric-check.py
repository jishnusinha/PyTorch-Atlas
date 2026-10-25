import math
import torch
x=torch.tensor(.7,dtype=torch.float32,requires_grad=True)
z=torch.sin(x**2); z.backward()
analytic=2*.7*math.cos(.7**2)
h=1e-3
finite=(math.sin((.7+h)**2)-math.sin((.7-h)**2))/(2*h)
assert abs(x.grad.item()-analytic)<1e-6
assert abs(x.grad.item()-finite)<2e-4
print(x.grad.item(),finite)
