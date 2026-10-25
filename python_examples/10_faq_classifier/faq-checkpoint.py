import tempfile
from pathlib import Path
import torch
from torch import nn

model = nn.Linear(3, 2)
vocab = {'<pad>': 0, '<unk>': 1, 'hello': 2}
answer_by_id = {0: 'Paris', 1: 'Tokyo'}
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'faq.pt'
    torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'answer_by_id': answer_by_id}, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    restored = nn.Linear(3, 2)
    restored.load_state_dict(saved['model_state'])
    assert saved['vocab'] == vocab and saved['answer_by_id'] == answer_by_id
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
print('weights and text mappings restored')
