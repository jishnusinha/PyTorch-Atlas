import torch

logits = torch.tensor([[0.1, -0.5, 1.8, 0.7]], dtype=torch.float32)
next_id = torch.argmax(logits, dim=1)
assert next_id.shape == (1,) and next_id.item() == 2
print("next token ID", next_id.item())
