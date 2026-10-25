import tempfile
from pathlib import Path
import torch
import torch.nn as nn

class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
x=torch.ones((4,30),dtype=torch.float32)
model.train()
training_output=model(x)
model.eval()
with torch.inference_mode():
    evaluation_output=model(x)
assert model.training is False and evaluation_output.shape==(4,1)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/"weights.pt"
    torch.save(model.state_dict(),path)
    restored=BinaryNet()
    state=torch.load(path,map_location="cpu",weights_only=True)
    restored.load_state_dict(state,strict=True)
    restored.eval()
    with torch.inference_mode():
        restored_output=restored(x)
    assert torch.equal(evaluation_output,restored_output)
print("state_dict round-trip:",tuple(restored_output.shape))
