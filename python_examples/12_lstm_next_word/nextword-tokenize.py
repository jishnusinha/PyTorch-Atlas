import re
import torch

def tokenize(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())

tokens = tokenize("Python libraries, for data science!")
assert tokens == ["python", "libraries", ",", "for", "data", "science", "!"]
assert torch.tensor([len(tokens)], dtype=torch.long).item() == 7
print(tokens)
