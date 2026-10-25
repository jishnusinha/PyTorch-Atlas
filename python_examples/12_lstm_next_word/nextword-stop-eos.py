import torch

EOS = 2
predictions = [5, 7, EOS, 4]
generated = []
for token_id in predictions:
    if token_id == EOS:
        break
    generated.append(token_id)
assert generated == [5, 7]
print("generated before EOS", generated)
