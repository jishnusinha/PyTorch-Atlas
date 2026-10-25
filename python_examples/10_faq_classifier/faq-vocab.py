import re
import torch

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())
def build_vocab(training_questions):
    vocab = {'<pad>': 0, '<unk>': 1}
    for text in training_questions:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
vocab = build_vocab(['Capital of France?', 'Red planet'])
encoded = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokenize('Capital of Mars?')], dtype=torch.long)
assert vocab['<pad>'] == 0 and vocab['<unk>'] == 1
assert encoded[-1].item() == vocab['<unk>']
print('vocabulary:', len(vocab), 'encoded:', encoded.tolist())
