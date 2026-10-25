import io
import torch
from torch import nn

torch.manual_seed(29)
first = nn.Linear(4, 2)
torch.manual_seed(29)
second = nn.Linear(4, 2)
assert torch.equal(first.weight, second.weight)
buffer = io.BytesIO()
torch.save(first.state_dict(), buffer)
buffer.seek(0)
restored = nn.Linear(4, 2)
restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
assert torch.equal(first.weight, restored.weight)
print("seeded initialization and weights-only checkpoint round-trip passed")
