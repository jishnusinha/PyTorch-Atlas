import torch

answer_tokens = torch.tensor([[7, 9, 12]], dtype=torch.long)
source_target = answer_tokens[0]
single_token_target = source_target[0].reshape(1)
answer_class_target = torch.tensor([2], dtype=torch.long)
assert source_target.shape == (3,) and single_token_target.shape == (1,)
assert answer_class_target.shape == (1,) and answer_class_target.dtype == torch.long
print('source first-token target:', single_token_target.tolist(), 'FAQ class target:', answer_class_target.tolist())
