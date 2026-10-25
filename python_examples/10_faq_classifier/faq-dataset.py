import re
import torch
from torch.utils.data import Dataset

def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())
class FAQDataset(Dataset):
    def __init__(self, rows, vocab, answer_to_id):
        self.rows, self.vocab, self.answer_to_id = rows, vocab, answer_to_id
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        question, answer = self.rows[index]
        ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokenize(question)]
        if not ids:
            ids = [self.vocab['<unk>']]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.answer_to_id[answer], dtype=torch.long)
rows = [('Capital of France?', 'geography')]
vocab = {'<pad>': 0, '<unk>': 1, 'capital': 2, 'of': 3, 'france': 4}
dataset = FAQDataset(rows, vocab, {'geography': 0})
question_ids, target = dataset[0]
assert question_ids.dtype == torch.long and target.shape == () and target.item() == 0
print(question_ids.tolist(), target.item())
