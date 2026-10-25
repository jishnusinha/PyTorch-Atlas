import torch

PAD = 0
rows = [[4, 8], [5, 6, 7]]
width = 4
padded = [([PAD] * (width - len(row)) + row) for row in rows]
matrix = torch.tensor(padded, dtype=torch.long)
x, y = matrix[:, :-1], matrix[:, -1]
assert matrix.tolist() == [[0, 0, 4, 8], [0, 5, 6, 7]]
assert x.shape == (2, 3) and y.shape == (2,)
assert y.tolist() == [8, 7]
print("X", tuple(x.shape), "y", tuple(y.shape))
