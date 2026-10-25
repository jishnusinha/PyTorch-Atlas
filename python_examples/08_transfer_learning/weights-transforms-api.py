import numpy as np
from PIL import Image
from torchvision.models import VGG16_Weights

weights = VGG16_Weights.DEFAULT
transform = weights.transforms()
image = Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8), mode='RGB')
x = transform(image)
assert x.shape == (3, 224, 224) and x.dtype.is_floating_point
print(tuple(x.shape), x.dtype)
print(transform)
