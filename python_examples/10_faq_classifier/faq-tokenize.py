import re
import torch

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())
tokens = tokenize('What is the capital of France?')
assert tokens == ['what', 'is', 'the', 'capital', 'of', 'france']
print(tokens)
