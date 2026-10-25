import torch

logits = torch.tensor([-8., 0., 8.], dtype=torch.float32)
gate = torch.sigmoid(logits)
candidate = torch.tanh(logits)
assert torch.all((gate >= 0) & (gate <= 1))
assert torch.all((candidate >= -1) & (candidate <= 1))
assert 0 < gate[0] < 1 and 0 < gate[-1] < 1
print("sigmoid:", gate.tolist(), "tanh:", candidate.tolist())
