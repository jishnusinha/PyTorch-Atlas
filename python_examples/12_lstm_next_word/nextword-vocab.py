from collections import Counter
import torch

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
tokens = ["python", "libraries", "python", "tools"]
counts = Counter(tokens)
vocab = {PAD: 0, UNK: 1, EOS: 2}
for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
    if word not in vocab:
        vocab[word] = len(vocab)
encoded = [vocab.get(word, vocab[UNK]) for word in ["python", "mystery"]]
assert encoded[1] == vocab[UNK] and vocab[PAD] != vocab[UNK]
print(vocab, encoded)
