import torch
from torch.nn.utils.rnn import pad_sequence

def collate_faq(batch, pad_id=0):
    questions, labels = zip(*batch)
    lengths = torch.tensor([len(q) for q in questions], dtype=torch.long)
    tokens = pad_sequence(questions, batch_first=True, padding_value=pad_id)
    return tokens, lengths, torch.stack(labels)
batch = [(torch.tensor([2, 3, 4]), torch.tensor(1)), (torch.tensor([5]), torch.tensor(0))]
tokens, lengths, labels = collate_faq(batch)
assert tokens.tolist() == [[2, 3, 4], [5, 0, 0]]
assert lengths.tolist() == [3, 1] and labels.shape == (2,)
print(tokens.shape, lengths.tolist(), labels.tolist())
