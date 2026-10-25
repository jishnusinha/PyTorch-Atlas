import torch
import torch.nn.functional as F

logits=torch.tensor([[-2.0],[0.0],[2.0]],dtype=torch.float32)
targets=torch.tensor([[0.0],[1.0],[1.0]],dtype=torch.float32)
probabilities=torch.sigmoid(logits)
epsilon=1e-7
p=probabilities.clamp(epsilon,1-epsilon)
manual=-(targets*torch.log(p)+(1-targets)*torch.log(1-p)).mean()
stable=F.binary_cross_entropy_with_logits(logits,targets)
assert manual.ndim==0 and torch.allclose(manual,stable,atol=1e-6)
print(f"manual={manual.item():.6f}, logits_loss={stable.item():.6f}")
