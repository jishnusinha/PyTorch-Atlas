import torch
from torch import nn
from pathlib import Path

model = nn.Linear(3, 4)
artifact = {"model_state": model.state_dict(), "vocab": {"<pad>": 0, "<unk>": 1},
            "config": {"input_dim": 3, "vocab_size": 4}}
path = Path("tmp-nextword-checkpoint.pt")
torch.save(artifact, path)
loaded = torch.load(path, map_location="cpu", weights_only=True)
assert loaded["vocab"]["<unk>"] == 1 and loaded["config"]["vocab_size"] == 4
path.unlink()
print("weights and preprocessing metadata round-trip")
