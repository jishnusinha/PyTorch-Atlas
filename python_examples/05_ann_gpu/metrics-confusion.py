import torch

confusion = torch.tensor([[8, 2, 0], [1, 3, 1], [0, 2, 3]], dtype=torch.float64)
true_positive = confusion.diag()
support = confusion.sum(dim=1)
predicted = confusion.sum(dim=0)
recall = true_positive / support.clamp_min(1)
precision = true_positive / predicted.clamp_min(1)
f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
accuracy = true_positive.sum() / confusion.sum()
balanced_accuracy = recall.mean()
macro_f1 = f1.mean()
assert confusion.sum().item() == 20
assert 0 <= accuracy <= 1 and 0 <= balanced_accuracy <= 1 and 0 <= macro_f1 <= 1
print(f"accuracy={accuracy:.3f}, balanced_accuracy={balanced_accuracy:.3f}, macro_f1={macro_f1:.3f}")
