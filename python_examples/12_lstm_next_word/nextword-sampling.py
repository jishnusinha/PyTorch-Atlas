import torch

torch.manual_seed(23)
logits = torch.tensor([2.0, 1.0, 0.0, -1.0], dtype=torch.float32)
temperature, k = 0.8, 3
values, ids = torch.topk(logits / temperature, k=k)
probabilities = torch.softmax(values, dim=0)
sampled = ids[torch.multinomial(probabilities, 1)]
assert probabilities.shape == (3,) and torch.isclose(probabilities.sum(), torch.tensor(1.0))
assert sampled.item() in ids.tolist()
print("top-k IDs", ids.tolist(), "probabilities", probabilities.tolist())
