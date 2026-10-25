import torch
from torch import nn
from torchvision.models import vgg16

model = vgg16(weights=None)
for parameter in model.features.parameters():
    parameter.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10))
trainable = [p for p in model.parameters() if p.requires_grad]
assert all(not p.requires_grad for p in model.features.parameters())
assert model.classifier[-1].out_features == 10
print('trainable tensors:', len(trainable), 'classes:', 10)
