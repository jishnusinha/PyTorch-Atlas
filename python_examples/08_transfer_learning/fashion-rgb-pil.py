import numpy as np
from PIL import Image

row = np.arange(784, dtype=np.int64) % 256
gray = row.reshape(28, 28).astype(np.uint8)
rgb = np.stack([gray] * 3, axis=-1)
image = Image.fromarray(rgb, mode='RGB')
assert rgb.shape == (28, 28, 3) and rgb.dtype == np.uint8
assert image.mode == 'RGB' and image.size == (28, 28)
assert np.array_equal(rgb[..., 0], rgb[..., 2])
print(rgb.shape, rgb.dtype, image.mode, image.size)
