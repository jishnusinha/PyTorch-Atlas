import torch

ids = torch.tensor([4, 8, 3, 9], dtype=torch.long)
pairs = [(ids[:end], ids[end]) for end in range(1, len(ids))]
assert pairs[0][0].tolist() == [4] and pairs[0][1].item() == 8
assert pairs[-1][0].tolist() == [4, 8, 3] and pairs[-1][1].item() == 9
assert all(target.item() not in x.tolist()[-1:] for x, target in pairs)
print([(x.tolist(), y.item()) for x, y in pairs])
