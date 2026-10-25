import torch
from torch import nn

torch.manual_seed(21)
class NextWord(nn.Module):
    def __init__(self, vocab_size=289, embedding_dim=100, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _sequence, (h_n, _c_n) = self.lstm(embedded)
        return self.head(h_n[-1])

model = NextWord()
logits = model(torch.ones((32, 61), dtype=torch.long))
assert logits.shape == (32, 289)
print("[32,61] -> [32,61,100] -> [32,61,150] ->", tuple(logits.shape))
