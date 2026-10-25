import torch

probabilities=torch.tensor([[.9],[.7],[.4],[.1]],dtype=torch.float32)
targets=torch.tensor([[1.],[0.],[1.],[0.]],dtype=torch.float32)
predicted=(probabilities>=0.5).to(torch.float32)
tp=((predicted==1)&(targets==1)).sum()
tn=((predicted==0)&(targets==0)).sum()
fp=((predicted==1)&(targets==0)).sum()
fn=((predicted==0)&(targets==1)).sum()
accuracy=(predicted==targets).to(torch.float32).mean()
assert predicted.shape==targets.shape==(4,1)
assert (tp.item(),tn.item(),fp.item(),fn.item())==(1,1,1,1)
print(f"threshold=.5 · TP={tp.item()} TN={tn.item()} FP={fp.item()} FN={fn.item()} · accuracy={accuracy.item():.2f}")
