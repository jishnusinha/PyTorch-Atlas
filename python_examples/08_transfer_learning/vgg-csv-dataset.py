import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FashionRows(Dataset):
    def __init__(self, features, labels, transform):
        self.features, self.labels, self.transform = features, labels, transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        pixels = np.asarray(self.features[index]).reshape(28, 28).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(np.stack([pixels] * 3, axis=-1), mode='RGB')
        return self.transform(image), torch.tensor(int(self.labels[index]), dtype=torch.long)
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
dataset = FashionRows(np.zeros((2, 784)), [3, 7], transform)
x, y = dataset[0]
assert x.shape == (3, 224, 224) and y.dtype == torch.long
print(tuple(x.shape), y.item())
