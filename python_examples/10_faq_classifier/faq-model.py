import torch
from torch import nn

class FAQClassifier(nn.Module):
    def __init__(self, vocab_size, num_answers, embed_dim=8, hidden_size=12, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, num_answers)
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        _, hidden = self.rnn(embedded)
        return self.output(hidden[-1])
model = FAQClassifier(vocab_size=20, num_answers=4)
logits = model(torch.tensor([[2, 3, 4], [5, 6, 0]], dtype=torch.long))
assert logits.shape == (2, 4) and torch.isfinite(logits).all()
print(logits.shape)
