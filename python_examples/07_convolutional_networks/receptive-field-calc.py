import torch

def add_layer(receptive_field, jump, kernel, stride):
    return receptive_field + (kernel - 1) * jump, jump * stride

r, j = 1, 1
trace = []
for _ in range(2):
    r, j = add_layer(r, j, 3, 1)
    r, j = add_layer(r, j, 3, 1)
    r, j = add_layer(r, j, 2, 2)
    trace.append((r, j))
assert trace == [(6, 2), (16, 4)]
print("after block 1:", trace[0], "after block 2:", trace[1])
