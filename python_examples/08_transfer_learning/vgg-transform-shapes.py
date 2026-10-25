import numpy as np
from PIL import Image
from torchvision import transforms

image = Image.fromarray(np.full((28, 28, 3), 128, dtype=np.uint8), mode='RGB')
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
x = transform(image)
assert x.shape == (3, 224, 224) and x.dtype.is_floating_point
assert x.min() >= 0 and x.max() <= 1
print(tuple(x.shape), x.dtype, x.min().item(), x.max().item())
