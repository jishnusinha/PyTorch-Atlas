import re
import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(27)
torch.set_num_threads(1)
rows = [
    ('What is the capital of France?', 'france_capital'),
    ('Name the capital city of France', 'france_capital'),
    ('What is the capital of Japan?', 'japan_capital'),
    ('Which city is Japan capital?', 'japan_capital'),
    ('Which planet is called the red planet?', 'red_planet'),
    ('Name the red planet', 'red_planet'),
    ('How many planets orbit the Sun?', 'planet_count'),
    ('Count planets in our solar system', 'planet_count'),
    ('Who created Python?', 'python_creator'),
    ('Who developed the Python language?', 'python_creator'),
    ('What does CPU stand for?', 'cpu_expansion'),
    ('Expand the abbreviation CPU', 'cpu_expansion'),
]
answer_text = {'france_capital': 'Paris', 'japan_capital': 'Tokyo', 'red_planet': 'Mars', 'planet_count': 'Eight', 'python_creator': 'Guido van Rossum', 'cpu_expansion': 'Central Processing Unit'}
answer_to_id = {name: i for i, name in enumerate(answer_text)}
train_rows = rows[::2]
valid_rows = rows[1::2]
def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())
vocab = {'<pad>': 0, '<unk>': 1}
for question, _ in train_rows:
    for token in tokenize(question):
        if token not in vocab:
            vocab[token] = len(vocab)
class FAQDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        question, answer = self.data[i]
        ids = [vocab.get(t, vocab['<unk>']) for t in tokenize(question)]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(answer_to_id[answer], dtype=torch.long)
def collate(batch):
    questions, targets = zip(*batch)
    lengths = torch.tensor([len(q) for q in questions], dtype=torch.long)
    tokens = pad_sequence(questions, batch_first=True, padding_value=vocab['<pad>'])
    return tokens, lengths, torch.stack(targets)
class FAQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 8, padding_idx=vocab['<pad>'])
        self.rnn = nn.RNN(8, 12, batch_first=True)
        self.head = nn.Linear(12, len(answer_to_id))
    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.head(hidden[-1])
train_loader = DataLoader(FAQDataset(train_rows), batch_size=4, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(27))
valid_loader = DataLoader(FAQDataset(valid_rows), batch_size=4, shuffle=False, collate_fn=collate)
model = FAQModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(35):
    model.train()
    for tokens, lengths, labels in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
model.eval()
validation_losses = []

with torch.inference_mode():
    for tokens, lengths, labels in valid_loader:
        validation_losses.append(loss_fn(model(tokens, lengths), labels).item())
assert validation_losses and all(torch.isfinite(torch.tensor(validation_losses)))
with tempfile.TemporaryDirectory() as folder:
    checkpoint = Path(folder) / 'faq.pt'
    torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'answer_to_id': answer_to_id, 'answer_text': answer_text}, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    restored = FAQModel()
    restored.load_state_dict(saved['model_state'])
    example = 'What is the capital of France?'
    ids = [saved['vocab'].get(t, saved['vocab']['<unk>']) for t in tokenize(example)]
    restored.eval()
    with torch.inference_mode():
        logits = restored(torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor([len(ids)]))
        probabilities = logits.softmax(dim=1)
        answer_id = probabilities.argmax(dim=1).item()
    assert logits.shape == (1, len(answer_to_id))
    assert saved['answer_text'][next(k for k, v in saved['answer_to_id'].items() if v == answer_id)]
print('vocab:', len(vocab), 'classes:', len(answer_to_id), 'validation loss:', round(sum(validation_losses) / len(validation_losses), 3))
