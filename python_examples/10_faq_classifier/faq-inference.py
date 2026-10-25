import re
import torch
from torch import nn

def predict_answer(model, text, vocab, answer_by_id, threshold=0.5):
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if not tokens:
        return "I don't know", 0.0
    unknown_only = all(token not in vocab for token in tokens)
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    model.eval()
    with torch.inference_mode():
        tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        logits = model(tokens, lengths)
        probabilities = logits.softmax(dim=1).squeeze(0)
        confidence, answer_id = probabilities.max(dim=0)
    if unknown_only or confidence.item() < threshold:
        return "I don't know", confidence.item()
    return answer_by_id[answer_id.item()], confidence.item()

assert callable(predict_answer)
print('Use a trained classifier and a calibrated threshold before inference.')
