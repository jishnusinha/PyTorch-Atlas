import torch

class MySimpleNN:
    def __init__(self,features=30):
        self.weights=(torch.randn((features,1),dtype=torch.float32)*0.05).requires_grad_()
        self.bias=torch.zeros((1,),dtype=torch.float32,requires_grad=True)
    def forward(self,x):
        logits=x@self.weights+self.bias
        return logits,torch.sigmoid(logits)
    def __call__(self,x):
        return self.forward(x)

model=MySimpleNN()
x=torch.ones((4,30),dtype=torch.float32)
logits,probabilities=model(x)
assert model.weights.shape==(30,1) and model.bias.shape==(1,)
assert logits.shape==(4,1) and probabilities.shape==(4,1)
assert probabilities.min()>=0 and probabilities.max()<=1
print("weights",tuple(model.weights.shape),"logits",tuple(logits.shape))
