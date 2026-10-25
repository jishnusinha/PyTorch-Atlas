import torch

batch_losses = [torch.tensor(0.4), torch.tensor(1.0)]
batch_sizes = [4, 1]
weighted = sum(loss.item() * size for loss, size in zip(batch_losses, batch_sizes)) / sum(batch_sizes)
unweighted = sum(loss.item() for loss in batch_losses) / len(batch_losses)
assert abs(weighted - 0.52) < 1e-6
assert abs(unweighted - 0.70) < 1e-6
print(f"example-weighted={weighted:.2f}, misleading batch mean={unweighted:.2f}")
