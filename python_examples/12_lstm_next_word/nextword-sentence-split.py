import random
import torch

sentences = ["a b c", "d e f", "g h i", "j k l", "m n o", "p q r"]
indices = list(range(len(sentences)))
random.Random(42).shuffle(indices)
validation_ids = set(indices[:2])
train = [sentence for i, sentence in enumerate(sentences) if i not in validation_ids]
validation = [sentence for i, sentence in enumerate(sentences) if i in validation_ids]
assert len(train) == 4 and len(validation) == 2
assert set(train).isdisjoint(validation)
print("train sentences", len(train), "validation sentences", len(validation))
