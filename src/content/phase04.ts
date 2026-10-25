import type { Lesson, Snippet } from "./types";

const makeSnippet = (
  id: string,
  title: string,
  purpose: string,
  shapes: string,
  code: string,
  expected: string,
): Snippet => ({
  id,
  title,
  purpose,
  dependencies: "Python 3.10+ · PyTorch · CPU · no downloads",
  shapes,
  code,
  expected,
  lines: code.split("\n").map((line, i) => {
    const trimmed = line.trim();
    const explanation = trimmed.startsWith("import ")
      ? "Import the dependency used by this example."
      : trimmed.startsWith("class ")
        ? "Define a reusable PyTorch or data-pipeline type."
        : trimmed.startsWith("def ")
          ? "Define one explicit operation in the example."
          : trimmed.startsWith("assert ")
            ? "Verify a shape, dtype, registration or numerical invariant."
            : trimmed.startsWith("print(")
              ? "Display the result stated below."
              : trimmed.startsWith("#")
                ? trimmed.replace(/^#+\s*/, "")
                : `Follow this operation through ${shapes}.`;
    return `Line ${i + 1} · ${explanation}`;
  }),
});

const snippets: Snippet[] = [
  makeSnippet(
    "manual-vs-idiomatic",
    "One update by hand and with SGD",
    "Start both implementations from the same scalar parameter and verify that torch.optim.SGD performs the same update.",
    "x [2,1], y [2,1], w [1,1], b [1]",
    `import torch
import torch.nn as nn

torch.manual_seed(4)
x=torch.tensor([[1.0],[2.0]],dtype=torch.float32)
y=torch.tensor([[0.0],[1.0]],dtype=torch.float32)
w=torch.tensor([[0.2]],requires_grad=True)
b=torch.tensor([0.1],requires_grad=True)
layer=nn.Linear(1,1)
with torch.no_grad():
    layer.weight.copy_(w)
    layer.bias.copy_(b)
manual_loss=nn.functional.mse_loss(x@w+b,y)
manual_loss.backward()
manual_w=w.detach()-0.05*w.grad
model_loss=nn.functional.mse_loss(layer(x),y)
opt=torch.optim.SGD(layer.parameters(),lr=0.05)
opt.zero_grad(set_to_none=True)
model_loss.backward()
opt.step()
assert torch.allclose(manual_w,layer.weight)
assert torch.allclose(w.grad,layer.weight.grad)
print(f"manual={manual_w.item():.4f}; SGD={layer.weight.item():.4f}")`,
    "manual=0.2350; SGD=0.2350 (same gradient update)",
  ),
  makeSnippet(
    "manual-class-vs-module",
    "Compare a plain model class with nn.Module",
    "Implement the same linear map with plain tensors and registered nn.Linear parameters; confirm equal outputs and different registration behavior.",
    "X [3,30], W [30,1], b [1] → output [3,1]",
    `import torch
import torch.nn as nn

class ManualModel:
    def __init__(self):
        self.weight=torch.nn.Parameter(torch.zeros((30,1)))
        self.bias=torch.nn.Parameter(torch.zeros(1))
    def __call__(self,x):
        return x@self.weight+self.bias
class RegisteredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
manual=ManualModel()
registered=RegisteredModel()
with torch.no_grad():
    registered.linear.weight.copy_(manual.weight.T)
    registered.linear.bias.copy_(manual.bias)
x=torch.ones((3,30),dtype=torch.float32)
assert torch.equal(manual(x),registered(x))
assert len(list(registered.parameters()))==2
assert not hasattr(manual,"parameters")
print("same output:",tuple(registered(x).shape),"; registered tensors:",len(list(registered.parameters())))`,
    "Both return [3,1]; nn.Module exposes its weight and bias through parameters()",
  ),
  makeSnippet(
    "idiomatic-classifier",
    "A stable 30-feature binary classifier",
    "Use a registered Linear layer, raw logits, BCEWithLogitsLoss and the standard optimizer lifecycle on deterministic synthetic rows.",
    "X [128,30] → logits [128,1]; y float32 [128,1]",
    `import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.manual_seed(12)
g=torch.Generator().manual_seed(12)
x=torch.randn((128,30),generator=g,dtype=torch.float32)
y=(x[:,0]-0.7*x[:,1]+0.4*x[:,2]>0).float().unsqueeze(1)
class BinaryNet(nn.Module):
    def __init__(self,n_features=30):
        super().__init__()
        self.linear=nn.Linear(n_features,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.15)
with torch.no_grad():
    initial=loss_fn(model(x),y).item()
for _ in range(30):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    loss=loss_fn(logits,y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.inference_mode():
    logits=model(x)
    accuracy=((logits>=0)==y.bool()).float().mean()
assert logits.shape==(128,1) and logits.dtype==torch.float32
assert loss.item()<initial and accuracy.item()>.8
print(f"loss {initial:.3f} -> {loss.item():.3f}; accuracy={accuracy.item():.2f}")`,
    "float32 logits [128,1]; training loss decreases; synthetic accuracy > 0.80",
  ),
  makeSnippet(
    "module-registration",
    "Register layers and inspect parameters",
    "Define a custom nn.Module, call it as model(x), inspect registered parameters and state_dict keys, and use a forward hook.",
    "input [4,30] → Linear logits [4,1]",
    `import torch
import torch.nn as nn

class BinaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
        self.temperature=nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        return self.linear(x)/self.temperature.clamp_min(0.1)
model=BinaryHead()
seen=[]
handle=model.linear.register_forward_hook(lambda module,args,out: seen.append(tuple(out.shape)))
x=torch.zeros((4,30),dtype=torch.float32)
logits=model(x)  # Module.__call__ invokes forward and its hooks
handle.remove()
keys=tuple(model.state_dict().keys())
names=tuple(name for name,_ in model.named_parameters())
assert logits.shape==(4,1) and seen==[(4,1)]
assert set(keys)=={"temperature","linear.weight","linear.bias"}
assert set(names)==set(keys)
assert sum(p.numel() for p in model.parameters())==32
print("registered:",", ".join(keys))`,
    "state_dict keys: temperature, linear.weight, linear.bias; 32 trainable scalars",
  ),
  makeSnippet(
    "module-hooks",
    "Inspect a module without replacing its forward method",
    "Attach and remove a forward hook around model(x); hooks observe outputs while normal module dispatch remains intact.",
    "module input [2,30] → hook output [2,8] → model output [2,1]",
    `import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(30,8)
        self.output=nn.Linear(8,1)
    def forward(self,x):
        return self.output(torch.relu(self.hidden(x)))
model=MLP()
observed=[]
hook=model.hidden.register_forward_hook(lambda m,inputs,out: observed.append(out.shape))
prediction=model(torch.ones((2,30)))
hook.remove()
assert observed==[torch.Size([2,8])]
assert prediction.shape==(2,1)
print("hook saw",tuple(observed[0]),"; prediction",tuple(prediction.shape))`,
    "hook saw (2, 8); prediction (2, 1)",
  ),
  makeSnippet(
    "sequential-summary",
    "Compare Sequential with explicit forward",
    "Build the same small MLP with nn.Sequential and named child modules; optionally inspect it with torchinfo when installed.",
    "X [5,30] → [5,8] → ReLU → logits [5,1]",
    `import importlib.util
import torch
import torch.nn as nn

sequential=nn.Sequential(nn.Linear(30,8),nn.ReLU(),nn.Linear(8,1))
class Explicit(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(30,8)
        self.output=nn.Linear(8,1)
    def forward(self,x):
        return self.output(torch.relu(self.hidden(x)))
model=Explicit()
x=torch.randn((5,30),dtype=torch.float32)
assert sequential(x).shape==model(x).shape==(5,1)
count=sum(p.numel() for p in model.parameters() if p.requires_grad)
assert count==257
if importlib.util.find_spec("torchinfo"):
    from torchinfo import summary
    summary(model,input_size=(5,30))
else:
    print("Optional torchinfo is not installed; parameter count:",count)
print("trainable parameters:",count)`,
    "output [5,1]; 257 trainable parameters; torchinfo is optional",
  ),
  makeSnippet(
    "loss-shapes",
    "Pair logits, probabilities and labels correctly",
    "Compare logits-first BCEWithLogitsLoss to Sigmoid plus BCELoss, and surface common target-shape and dtype errors.",
    "logits [3,1], probabilities [3,1], float targets [3,1]",
    `import torch
import torch.nn as nn

logits=torch.tensor([[-2.0],[0.0],[2.0]],dtype=torch.float32)
target=torch.tensor([[0.0],[1.0],[1.0]],dtype=torch.float32)
stable=nn.BCEWithLogitsLoss()(logits,target)
probability=logits.sigmoid()
classic=nn.BCELoss()(probability,target)
assert torch.allclose(stable,classic,atol=1e-6)
shape_error=False
try:
    nn.BCEWithLogitsLoss()(logits,target.squeeze(1))
except (ValueError,RuntimeError):
    shape_error=True
assert shape_error  # [B,1] logits do not pair with [B] labels
type_error=False
try:
    nn.BCEWithLogitsLoss()(logits,target.long())
except (RuntimeError,TypeError):
    type_error=True
assert type_error  # binary BCE targets should be floating point
print(f"both valid losses={stable.item():.4f}; mismatches caught")`,
    "The two valid formulations agree; [B] shape and integer target mismatches are caught",
  ),
  makeSnippet(
    "loss-error-repair",
    "Repair the prediction and target contract",
    "Normalize model outputs and binary targets to matching [B,1] float32 tensors before computing the loss.",
    "raw output [B] → logits [B,1]; labels [B] → float32 [B,1]",
    `import torch
import torch.nn as nn

raw_logits=torch.tensor([0.2,-0.4,1.1],dtype=torch.float32)
raw_labels=torch.tensor([0,1,1],dtype=torch.int64)
logits=raw_logits.reshape(-1,1)
target=raw_labels.to(torch.float32).reshape(-1,1)
loss=nn.BCEWithLogitsLoss()(logits,target)
assert logits.shape==target.shape==(3,1)
assert target.dtype==torch.float32 and torch.isfinite(loss)
print("repaired shapes:",tuple(logits.shape),tuple(target.shape))`,
    "logits and targets both [3,1] float32; finite scalar loss",
  ),
  makeSnippet(
    "optimizer-lifecycle",
    "Clear gradients, backpropagate and step",
    "Let SGD own parameter updates, clear accumulated gradients with set_to_none=True, and compare training and evaluation modes.",
    "X [16,2] → logits [16,1] → scalar loss",
    `import torch
import torch.nn as nn

torch.manual_seed(2)
x=torch.arange(32,dtype=torch.float32).reshape(16,2)/16
y=(x[:,0:1]+x[:,1:2]>.9).float()
model=nn.Linear(2,1)
optimizer=torch.optim.SGD(model.parameters(),lr=0.3)
loss_fn=nn.BCEWithLogitsLoss()
with torch.no_grad():
    initial=loss_fn(model(x),y).item()
for _ in range(25):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    loss=loss_fn(model(x),y)
    loss.backward()
    assert model.weight.grad is not None
    optimizer.step()
model.eval()
with torch.inference_mode():
    final=loss_fn(model(x),y).item()
assert final<initial
print(f"loss {initial:.3f} -> {final:.3f}")`,
    "SGD updates registered parameters; final loss is below initial loss",
  ),
  makeSnippet(
    "custom-dataset",
    "Implement a map-style Dataset",
    "Store paired feature and target tensors, then expose one sample through __getitem__ and the row count through __len__.",
    "features [12,30], labels [12,1] → item [30] + [1]",
    `import torch
from torch.utils.data import Dataset

class BreastRows(Dataset):
    def __init__(self,features,labels):
        if len(features)!=len(labels):
            raise ValueError("features and labels need the same row count")
        self.features=features.to(torch.float32)
        self.labels=labels.to(torch.float32).reshape(-1,1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
x=torch.arange(360,dtype=torch.float32).reshape(12,30)
y=(torch.arange(12)%2).float()
data=BreastRows(x,y)
row,label=data[3]
assert len(data)==12 and row.shape==(30,) and label.shape==(1,)
assert label.item()==1.0 and row.dtype==torch.float32
print("sample 3:",tuple(row.shape),tuple(label.shape),label.item())`,
    "12 rows; one feature vector [30] and paired label [1] float32",
  ),
  makeSnippet(
    "tensor-dataset",
    "Use TensorDataset for aligned tensors",
    "Use the built-in map-style dataset when tensors already share a leading sample dimension; reserve a custom Dataset for custom retrieval or transforms.",
    "X [10,30] and y [10,1] → paired item [30] + [1]",
    `import torch
from torch.utils.data import TensorDataset

x=torch.randn((10,30),generator=torch.Generator().manual_seed(1),dtype=torch.float32)
y=(x[:,0:1]>0).float()
dataset=TensorDataset(x,y)
features,target=dataset[4]
assert len(dataset)==10 and features.shape==(30,) and target.shape==(1,)
assert torch.equal(features,x[4]) and torch.equal(target,y[4])
print("TensorDataset pairs:",tuple(features.shape),tuple(target.shape))`,
    "ten paired records; sample shapes [30] and [1]",
  ),
  makeSnippet(
    "dataset-vs-dataloader",
    "Compare one lookup with one minibatch",
    "Read one paired item directly from TensorDataset, then wrap the same dataset in DataLoader to see how collation adds a batch axis.",
    "dataset[3]: [30]+[1]; loader batch: [4,30]+[4,1]",
    `import torch
from torch.utils.data import DataLoader,TensorDataset

x=torch.arange(300,dtype=torch.float32).reshape(10,30)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
one_x,one_y=dataset[3]
batch_x,batch_y=next(iter(DataLoader(dataset,batch_size=4,shuffle=False)))
assert one_x.shape==(30,) and one_y.shape==(1,)
assert batch_x.shape==(4,30) and batch_y.shape==(4,1)
assert torch.equal(batch_x[3],one_x) and batch_y[3].item()==one_y.item()
print("one sample",tuple(one_x.shape),tuple(one_y.shape),"; batch",tuple(batch_x.shape),tuple(batch_y.shape))`,
    "One item has no batch axis; DataLoader collation produces [4,30] and [4,1]",
  ),
  makeSnippet(
    "dataloader-pipeline",
    "Trace sampling, fetching and batch collation",
    "Make the map-style DataLoader pipeline explicit with a sampler, a deterministic dataset, and default tensor stacking.",
    "10 samples → sampler indices → batches [4,30] and [2,30]",
    `import torch
from torch.utils.data import DataLoader,SequentialSampler,TensorDataset

x=torch.arange(300,dtype=torch.float32).reshape(10,30)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
sampler=SequentialSampler(dataset)
loader=DataLoader(dataset,batch_size=4,sampler=sampler,drop_last=False)
batches=list(loader)
assert [len(batch_y) for _,batch_y in batches]==[4,4,2]
assert batches[0][1].flatten().tolist()==[0.,1.,2.,3.]
assert batches[-1][0].shape==(2,30)
print("batch sizes:",[len(labels) for _,labels in batches])`,
    "batch sizes: [4, 4, 2]; default collate stacks aligned rows",
  ),
  makeSnippet(
    "loader-options",
    "Choose train and evaluation loader options",
    "Shuffle training samples, preserve evaluation order, and inspect drop_last and pin-memory behavior with portable zero-worker loading.",
    "N=10, B=4 → train batches 4,4,2; drop_last → 4,4",
    `import torch
from torch.utils.data import DataLoader,RandomSampler,TensorDataset

x=torch.arange(20,dtype=torch.float32).reshape(10,2)
y=torch.arange(10,dtype=torch.float32).reshape(10,1)
dataset=TensorDataset(x,y)
train=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0,
                 drop_last=False,pin_memory=torch.cuda.is_available())
random_sampler=RandomSampler(dataset,generator=torch.Generator().manual_seed(8))
sampled=DataLoader(dataset,batch_size=4,sampler=random_sampler,num_workers=0)
evaluation=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=0)
dropped=DataLoader(dataset,batch_size=4,shuffle=False,drop_last=True,num_workers=0)
assert [len(by) for _,by in train]==[4,4,2]
assert [len(by) for _,by in evaluation]==[4,4,2]
assert [len(by) for _,by in dropped]==[4,4]
assert next(iter(evaluation))[1][0].item()==0.0
sampled_ids=torch.cat([by.flatten() for _,by in sampled]).long().tolist()
assert sorted(sampled_ids)==list(range(10))
print("train/eval keep 10 rows; drop_last keeps",sum(len(by) for _,by in dropped))`,
    "train/eval retain 10 samples; drop_last=True emits 8 this pass",
  ),
  makeSnippet(
    "text-collate",
    "Pad variable-length text in collate_fn",
    "Encode string labels explicitly and use a custom collate function to pad token lists into a rectangular batch.",
    "token rows lengths 2,4,3 → tokens [3,4], lengths [3], labels [3]",
    `import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset

class Sentences(Dataset):
    def __init__(self,rows):
        self.rows=rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        words,label=self.rows[index]
        return torch.tensor(words,dtype=torch.long),label
def collate_text(items):
    sequences,labels=zip(*items)
    padded=pad_sequence(sequences,batch_first=True,padding_value=0)
    lengths=torch.tensor([len(s) for s in sequences],dtype=torch.long)
    label_ids={"negative":0,"positive":1}
    targets=torch.tensor([label_ids[label] for label in labels],dtype=torch.long)
    return padded,lengths,targets
rows=[([4,5],"positive"),([7,8,9,10],"negative"),([2,3,4],"positive")]
tokens,lengths,labels=next(iter(DataLoader(Sentences(rows),batch_size=3,collate_fn=collate_text)))
assert tokens.shape==(3,4) and lengths.tolist()==[2,4,3]
assert labels.tolist()==[1,0,1] and tokens[0,2:].tolist()==[0,0]
print("padded tokens",tuple(tokens.shape),"lengths",lengths.tolist())`,
    "padded token IDs [3,4], lengths [2,4,3], encoded long labels [3]",
  ),
  makeSnippet(
    "windows-workers",
    "Keep data loading portable on Windows",
    "Keep num_workers=0 as the simplest default; guard worker creation behind a main check in standalone Windows scripts.",
    "Dataset samples → worker processes when enabled → batches",
    `import torch
from torch.utils.data import DataLoader,TensorDataset

def main():
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)
    y=torch.arange(8,dtype=torch.float32).reshape(8,1)
    dataset=TensorDataset(x,y)
    # Start with zero workers; test multiprocessing from a .py file, not a notebook cell.
    workers=0
    loader=DataLoader(dataset,batch_size=4,num_workers=workers)
    batches=list(loader)
    assert len(batches)==2 and batches[0][0].shape==(4,3)
    print(f"num_workers={workers}; batches={len(batches)}")

if __name__=="__main__":
    main()`,
    "num_workers=0; batches=2 (portable baseline)",
  ),
  makeSnippet(
    "batched-classifier",
    "Train a 30-feature classifier by mini-batch",
    "Use a deterministic synthetic split, fit standardization on training rows only, then train and evaluate a logits-first model with loaders.",
    "X [N,30] → batches [B,30] → logits/targets [B,1]",
    `import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
torch.manual_seed(22)
g=torch.Generator().manual_seed(22)
x=torch.randn((180,30),generator=g,dtype=torch.float32)
y=(x[:,0]-0.8*x[:,1]+0.5*x[:,2]>0).float().reshape(-1,1)
ids=torch.randperm(len(x),generator=g)
train_ids,val_ids,test_ids=ids[:120],ids[120:150],ids[150:]
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
train_loader=DataLoader(TensorDataset(x_train,y_train),batch_size=16,shuffle=True,generator=g)
val_loader=DataLoader(TensorDataset(x_val,y_val),batch_size=16,shuffle=False)
test_loader=DataLoader(TensorDataset(x_test,y_test),batch_size=16,shuffle=False)
class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.2)
def evaluate(loader):
    model.eval()
    total=correct=0
    with torch.inference_mode():
        for features,target in loader:
            logits=model(features)
            correct+=((logits>=0)==target.bool()).sum().item()
            total+=len(target)
    return correct/total
with torch.no_grad():
    initial=loss_fn(model(x_train),y_train).item()
for _ in range(12):
    model.train()
    for features,target in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(model(features),target)
        loss.backward()
        optimizer.step()
model.eval()
with torch.inference_mode():
    final=loss_fn(model(x_train),y_train).item()
val_accuracy=evaluate(val_loader)
test_accuracy=evaluate(test_loader)
assert x_train.shape[1]==30 and y_train.shape[1]==1
assert x_train.dtype==torch.float32 and final<initial
assert 0.0<=val_accuracy<=1.0 and 0.0<=test_accuracy<=1.0
assert val_accuracy>.75 and test_accuracy>.75
print(f"train loss {initial:.3f} -> {final:.3f}; val={val_accuracy:.2f}; test={test_accuracy:.2f}")`,
    "Train loss decreases; validation/test accuracy are computed in eval + inference_mode",
  ),
  makeSnippet(
    "state-dict-modes",
    "Evaluate and save weights safely",
    "Switch explicitly between train/eval modes and save only a state_dict to a temporary file, loading it on CPU with weights_only=True.",
    "model input [B,30] → logits [B,1]; state_dict contains parameters",
    `import tempfile
from pathlib import Path
import torch
import torch.nn as nn

class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(30,1)
    def forward(self,x):
        return self.linear(x)
model=BinaryNet()
x=torch.ones((4,30),dtype=torch.float32)
model.train()
training_output=model(x)
model.eval()
with torch.inference_mode():
    evaluation_output=model(x)
assert model.training is False and evaluation_output.shape==(4,1)
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/"weights.pt"
    torch.save(model.state_dict(),path)
    restored=BinaryNet()
    state=torch.load(path,map_location="cpu",weights_only=True)
    restored.load_state_dict(state,strict=True)
    restored.eval()
    with torch.inference_mode():
        restored_output=restored(x)
    assert torch.equal(evaluation_output,restored_output)
print("state_dict round-trip:",tuple(restored_output.shape))`,
    "eval logits [4,1] unchanged after a weights-only state_dict round-trip",
  ),
];

const get = (id: string) => {
  const snippet = snippets.find((item) => item.id === id);
  if (!snippet) throw new Error(`Missing Phase 04 snippet: ${id}`);
  return snippet;
};
function lesson(config: {
  id: string;
  moduleId: "training" | "data-pipeline";
  title: string;
  subtitle: string;
  source: [number, number];
  kind: "module" | "lifecycle" | "loader" | "collate" | "batch" | "pipeline";
  diagramTitle: string;
  diagramDescription: string;
  objective: string;
  intuition: string;
  equation: string;
  numerical: string;
  shapes: [string, string, string][];
  sections: [string, string, string][];
  snippet: string;
  template: string;
  additional?: string[];
  pitfalls: string[];
  production: string;
  revision: string[];
  questions: [string, string][];
}): Lesson {
  return {
    id: config.id,
    moduleId: config.moduleId,
    title: config.title,
    subtitle: config.subtitle,
    minutes: 8,
    source: { start: config.source[0], end: config.source[1], kind: "source-derived" },
    objectives: [
      config.objective,
      "Trace the associated tensor or parameter shapes.",
      "Choose a safe, idiomatic PyTorch pattern for the task.",
    ],
    intuition: config.intuition,
    equation: config.equation,
    numerical: config.numerical,
    shapes: config.shapes.map(([name, shape, meaning]) => ({ name, shape, meaning })),
    diagram: {
      kind: config.kind,
      title: config.diagramTitle,
      description: config.diagramDescription,
    },
    sections: config.sections.map(([id, title, paragraphs]) => ({
      id,
      title,
      paragraphs: paragraphs.split("\n"),
    })),
    snippet: get(config.snippet),
    template: get(config.template),
    additionalSnippets: config.additional?.map(get),
    pitfalls: config.pitfalls,
    production: config.production,
    revision: config.revision,
    questions: config.questions.map(([question, answer]) => ({ question, answer })),
  };
}

export const phase04Snippets = snippets;
export const phase04Lessons: Lesson[] = [
  lesson({
    id: "manual-vs-idiomatic",
    moduleId: "training",
    title: "From manual weights to registered modules",
    subtitle: "Keep the model mathematics; let PyTorch manage its parameters.",
    source: [15, 16],
    kind: "lifecycle",
    diagramTitle: "The same gradient, two update paths",
    diagramDescription:
      "A hand-written weight update and an SGD optimizer both consume the same gradient and reach the same updated weight.",
    objective:
      "Compare explicit tensor parameters and manual updates with registered layers and torch.optim.",
    intuition:
      "A manual implementation is a transparent sketch. An optimizer is a reliable tool that tracks the parameters it owns and applies a consistent update policy.",
    equation: "θnew = θold − η · ∂L/∂θ",
    numerical:
      "For a scalar weight 0.20, gradient 0.40 and learning rate 0.05, both paths produce 0.20 − 0.05×0.40 = 0.18.",
    shapes: [
      ["Features", "[B,1]", "One feature for each record."],
      ["Weight", "[1,1]", "One trainable coefficient."],
      ["Prediction", "[B,1]", "One output per row."],
    ],
    sections: [
      [
        "why",
        "What the framework takes over",
        "The Phase 03 classifier stores W and b directly, calculates loss and gradients, then updates parameters inside no_grad. PyTorch modules register trainable tensors; optimizers discover them through model.parameters() and update them after backward.\nThis changes who manages the bookkeeping, not the chain rule. The included scalar check starts both approaches from identical values and asserts their SGD update matches the manual equation.",
      ],
      [
        "when",
        "Choose the right level of abstraction",
        "Hand-written gradients are useful for learning and tiny derivations. For real models, registration, optimizer parameter groups, state dictionaries, hooks and standard loss modules reduce bookkeeping errors.\nThe source's Sigmoid plus BCELoss version is retained in the comparison. Later lessons use raw logits with BCEWithLogitsLoss, which is numerically more stable.",
      ],
    ],
    snippet: "manual-vs-idiomatic",
    template: "idiomatic-classifier",
    additional: ["manual-class-vs-module"],
    pitfalls: [
      "Do not pass unregistered plain tensors to an optimizer and expect it to find them.",
      "Do not manually update a parameter outside no_grad or an optimizer.",
      "Optimizer gradients accumulate just like leaf tensor gradients until cleared.",
    ],
    production:
      "Compare equivalent updates only when initialization, loss reduction, batch and learning rate match. Optimizers can add state or scheduling, so their behavior may go beyond plain SGD.",
    revision: [
      "nn.Module registers model state.",
      "model.parameters() supplies optimizer parameters.",
      "backward computes gradients; optimizer.step updates parameters.",
      "A manual update remains useful as a reference calculation.",
    ],
    questions: [
      [
        "What does an optimizer receive?",
        "Usually model.parameters(), which iterates registered trainable parameters.",
      ],
      [
        "Does SGD remove the need for backward?",
        "No. backward computes gradients; step applies the optimizer update.",
      ],
      [
        "When is a manual update still useful?",
        "For learning, debugging a small derivative, or checking an optimizer calculation.",
      ],
      [
        "What must match for a fair comparison?",
        "Starting parameters, objective/reduction, data batch and learning rate.",
      ],
    ],
  }),
  lesson({
    id: "nn-module",
    moduleId: "training",
    title: "Build a model with nn.Module",
    subtitle: "Register parameters, define forward, call the model object.",
    source: [16, 17],
    kind: "module",
    diagramTitle: "Registered state belongs to the module tree",
    diagramDescription:
      "A BinaryHead registers a Linear layer's weight and bias plus a custom Parameter; these appear in named_parameters and state_dict.",
    objective:
      "Implement an nn.Module with super().__init__(), child layers, a forward method, model(x), parameters and state_dict.",
    intuition:
      "A module is a labeled equipment case: its child components and trainable settings are inventoried together so optimizers and checkpoints can find them.",
    equation: "logits = model(x) = forward(x; registered parameters)",
    numerical:
      "A Linear(30,1) layer contains 30×1 weights and one bias, or 31 trainable scalars. Adding one scalar nn.Parameter gives 32.",
    shapes: [
      ["Input", "[B,30]", "Thirty scaled features for each record."],
      ["Weight", "[30,1]", "Registered Linear matrix."],
      ["Logits", "[B,1]", "One raw score per record."],
    ],
    sections: [
      [
        "contract",
        "Initialize the base class before assigning modules",
        "Subclass torch.nn.Module and call super().__init__() first. This initializes registration machinery. Assign an nn.Linear or nn.Parameter to self to register it; a local tensor or a layer hidden in an ordinary Python list is not automatically registered.\nWrite forward(self,x) for the tensor operations. Call model(x), not model.forward(x): Module.__call__ handles hooks and framework dispatch before it reaches forward.",
      ],
      [
        "inspection",
        "Inspect and observe registered state",
        "model.parameters() iterates registered tensors, named_parameters() gives their paths, and state_dict() maps parameter and buffer names to tensors. A forward hook can inspect an intermediate output and should be removed after use.\nThe parameter tree diagram corresponds to the executable BinaryHead: Linear(30,1) contributes 31 values and temperature contributes one more.",
      ],
    ],
    snippet: "module-registration",
    template: "module-hooks",
    pitfalls: [
      "Omitting super().__init__() prevents child modules from registering correctly.",
      "Calling model.forward(x) bypasses Module.__call__ hooks and dispatch.",
      "A tensor assigned without nn.Parameter is not a trainable registered parameter.",
      "Do not keep a temporary forward hook attached after inspection.",
    ],
    production:
      "Use named ModuleList or Sequential containers for variable numbers of child layers. Store fixed non-trainable tensors as buffers when they belong in device moves and checkpoints.",
    revision: [
      "super().__init__() establishes Module registration.",
      "Assign layers and nn.Parameter objects as attributes.",
      "Define forward; invoke the model with model(x).",
      "Inspect with named_parameters() and state_dict().",
    ],
    questions: [
      [
        "Why call super().__init__() first?",
        "It initializes the base Module so assigned parameters and child modules can be registered.",
      ],
      [
        "What is the difference between forward and model(x)?",
        "forward describes computation; model(x) invokes Module.__call__, which also handles hooks and framework behavior.",
      ],
      [
        "Where does nn.Linear(30,1) store its trainable values?",
        "As registered weight [1,30] and bias [1], 31 scalars total.",
      ],
      [
        "What does state_dict() contain?",
        "A name-to-tensor mapping for registered parameters and persistent buffers.",
      ],
    ],
  }),
  lesson({
    id: "sequential-architecture",
    moduleId: "training",
    title: "Stack layers or define the forward path",
    subtitle:
      "Use Sequential for a straight line; use explicit forward for branching logic.",
    source: [17, 18],
    kind: "module",
    diagramTitle: "A small MLP as registered stages",
    diagramDescription:
      "Thirty input features pass through Linear(30,8), ReLU and Linear(8,1), producing one logit per example.",
    objective:
      "Choose between nn.Sequential and named modules with an explicit forward method; inspect activation and parameter count.",
    intuition:
      "Sequential is a conveyor belt: each stage hands its output to the next. Explicit forward is a workbench where you can split, combine or reuse intermediate values.",
    equation: "[B,30] → Linear → [B,8] → ReLU → Linear → [B,1]",
    numerical:
      "The example has (30×8+8) + (8×1+1) = 257 trainable parameters. ReLU adds no parameters.",
    shapes: [
      ["Input", "[B,30]", "Batch of 30-feature rows."],
      ["Hidden activation", "[B,8]", "Eight learned features per row."],
      ["Output", "[B,1]", "Raw binary-classification logit."],
    ],
    sections: [
      [
        "straight",
        "Sequential makes a simple chain readable",
        "nn.Sequential(nn.Linear(30,8), nn.ReLU(), nn.Linear(8,1)) registers each stage in order and forwards values through them. ReLU is nonlinear; stacking only linear layers would still be one linear transform.\nThe final layer emits a raw logit. Do not append Sigmoid when the training objective is BCEWithLogitsLoss.",
      ],
      [
        "flexible",
        "Use explicit forward when data flow branches",
        "Named submodules and an explicit forward are clearer when the model reuses a layer, adds a residual, combines multiple inputs, or returns intermediate outputs. Both styles register their layers and expose parameters.\nThe source suggests torchinfo.summary(model,input_size=(10,5)). torchinfo is an optional third-party inspection aid; count parameters directly when it is unavailable, and provide an input shape with the correct feature width.",
      ],
    ],
    snippet: "sequential-summary",
    template: "idiomatic-classifier",
    pitfalls: [
      "Linear layers transform the last dimension; batch dimension is preserved.",
      "ReLU contributes nonlinearity but has no trainable parameters.",
      "torchinfo is optional and must not become a required runtime dependency.",
      "A summary input shape must match the model's feature count.",
    ],
    production:
      "Prefer an explicit class once branching, masking, residuals, custom validation, or multiple outputs make a layer list hard to understand.",
    revision: [
      "Sequential describes a simple straight chain.",
      "Explicit forward describes flexible data flow.",
      "Activations add nonlinear transformations.",
      "Parameter count sums weight and bias elements; ReLU adds none.",
    ],
    questions: [
      [
        "When is Sequential a good fit?",
        "When each stage feeds directly into the next in one straight sequence.",
      ],
      ["What is the output shape of Linear(30,8) on [B,30]?", "[B,8]."],
      ["How many parameters are in Linear(30,8)?", "30×8 weights plus 8 biases = 248."],
      [
        "Why use ReLU between two linear layers?",
        "Without a nonlinear activation, the composition remains a single linear transformation.",
      ],
    ],
  }),
  lesson({
    id: "loss-conventions",
    moduleId: "training",
    title: "Match logits, loss and target shapes",
    subtitle: "Keep binary targets paired with one raw logit per example.",
    source: [18, 19],
    kind: "module",
    diagramTitle: "The binary loss contract",
    diagramDescription:
      "Raw logits and float targets both have shape [B,1]; BCEWithLogitsLoss combines sigmoid and binary cross entropy stably.",
    objective:
      "Distinguish BCELoss probabilities from BCEWithLogitsLoss raw logits and repair batch shape and dtype mismatches.",
    intuition:
      "The loss is a measuring instrument calibrated for a particular signal. Give a logits-based loss raw logits, not already-squashed probabilities.",
    equation: "BCEWithLogits(z,y) = BCE(sigmoid(z),y)",
    numerical:
      "For logits [-2,0,2] and targets [0,1,1], BCEWithLogitsLoss matches BCELoss(sigmoid(logits),targets) within floating-point tolerance.",
    shapes: [
      ["Logits", "[B,1]", "Unbounded raw scores."],
      ["Probability alternative", "[B,1]", "Sigmoid(logits) for BCELoss."],
      ["Targets", "[B,1] float32", "Binary values 0.0 or 1.0."],
    ],
    sections: [
      [
        "pair",
        "Use the loss that matches model output",
        "BCELoss consumes probabilities in [0,1], so the model must apply Sigmoid first. BCEWithLogitsLoss consumes raw logits and combines sigmoid with cross entropy using a stable formulation; this is the recommended default for binary classification.\nThese formulations express the same objective for the same examples and reduction, but computing from logits avoids a separate probability operation in the loss path.",
      ],
      [
        "shapes",
        "Make each target line up with one prediction",
        "The source encountered prediction [455,1] with target [455]. Reshape labels to [B,1] (or deliberately squeeze predictions to [B]) so shapes exactly match. For BCE, targets should be floating-point values, not integer class IDs.\nUse CrossEntropyLoss instead for mutually exclusive multi-class labels: its logits are [B,C] and class-index targets are integer [B].",
      ],
    ],
    snippet: "loss-shapes",
    template: "loss-error-repair",
    pitfalls: [
      "Do not pass sigmoid probabilities into BCEWithLogitsLoss.",
      "Do not pass raw logits to BCELoss.",
      "A [B,1] versus [B] mismatch can broadcast or error; make the contract explicit.",
      "Binary BCE targets should be float32; class-index CrossEntropy targets should be long.",
    ],
    production:
      "Assert logits.shape == target.shape at the loss boundary. Choose loss reduction deliberately when comparing batch sizes or aggregating metrics.",
    revision: [
      "BCELoss expects probabilities.",
      "BCEWithLogitsLoss expects raw logits and is the stable default.",
      "Binary BCE target is float and matches [B,1] logits.",
      "CrossEntropyLoss uses [B,C] logits and long [B] class IDs.",
    ],
    questions: [
      [
        "What should feed BCEWithLogitsLoss?",
        "Raw, unnormalized logits, with float targets of the same shape.",
      ],
      [
        "What shape mismatch did the source report?",
        "Predictions [455,1] and targets [455]; reshape targets to [455,1] or consistently squeeze the predictions.",
      ],
      [
        "Does BCELoss consume raw logits?",
        "No. BCELoss consumes probabilities, typically after sigmoid.",
      ],
      [
        "What target dtype does CrossEntropyLoss use for class indices?",
        "torch.long with shape [B].",
      ],
    ],
  }),
  lesson({
    id: "optimizer-lifecycle",
    moduleId: "training",
    title: "Let the optimizer own each update",
    subtitle: "Clear, differentiate, step—and keep training separate from evaluation.",
    source: [19, 20],
    kind: "lifecycle",
    diagramTitle: "One optimizer update, in order",
    diagramDescription:
      "A training batch flows through zero_grad(set_to_none=True), forward, scalar loss, backward and optimizer.step; evaluation switches to eval and inference_mode without a step.",
    objective:
      "Use torch.optim.SGD and the zero_grad, backward, step lifecycle without accidentally accumulating gradients.",
    intuition:
      "Gradients are a running notepad. Clear it before a new batch, write this batch's derivative, then let the optimizer apply the update.",
    equation: "clear → forward → loss → backward → step",
    numerical:
      "For a scalar objective with starting loss 0.69, repeated SGD steps lower the toy training loss; each optimizer.step uses the gradient just computed for that batch.",
    shapes: [
      ["Batch", "[B,30]", "The current training minibatch."],
      [
        "Parameter gradient",
        "same as parameter",
        "One derivative per registered trainable value.",
      ],
      ["Loss", "[]", "A scalar objective for backward."],
    ],
    sections: [
      [
        "train",
        "One update per training batch",
        "Construct an optimizer from model.parameters(). A conventional loop sets model.train(), clears stale gradients with optimizer.zero_grad(set_to_none=True), computes model(x), measures a scalar loss, calls backward and then optimizer.step().\nThe source clears gradients immediately before backward. That works because gradients are only needed for the current backward call; clearing before forward is a common loop pattern that also makes the boundary explicit.",
      ],
      [
        "evaluate",
        "Evaluation changes two separate switches",
        "model.eval() changes modules such as Dropout and BatchNorm. torch.no_grad() or torch.inference_mode() disables gradient recording. Evaluation usually uses both and never calls optimizer.step().\nGradient clearing matters because .grad accumulates by default. set_to_none=True sets gradients to None and can avoid unnecessary writes; optimizers treat missing gradients as parameters to skip.",
      ],
    ],
    snippet: "optimizer-lifecycle",
    template: "manual-vs-idiomatic",
    pitfalls: [
      "Calling backward twice without clearing accumulates gradients.",
      "optimizer.zero_grad() does not update parameters; optimizer.step() does.",
      "model.eval() does not disable autograd by itself.",
      "no_grad() does not switch Dropout or BatchNorm into evaluation mode.",
    ],
    production:
      "For validation, use model.eval() with inference_mode(); return to model.train() before the next training epoch. Never choose a checkpoint by repeatedly tuning on the final test set.",
    revision: [
      "Train mode controls train/eval-specific module behavior.",
      "zero_grad clears old gradients.",
      "backward calculates this batch's gradients.",
      "step updates parameters; evaluation has no step.",
    ],
    questions: [
      [
        "What order should one training batch follow?",
        "Clear gradients, run forward, calculate loss, call backward, then optimizer.step().",
      ],
      ["Why clear gradients each batch?", "Autograd accumulates into .grad by default."],
      [
        "Does eval() disable gradient recording?",
        "No. Pair it with no_grad() or inference_mode() when evaluating.",
      ],
      [
        "What does set_to_none=True do?",
        "It leaves .grad as None until a new gradient is produced and lets the optimizer skip parameters without gradients.",
      ],
    ],
  }),
  lesson({
    id: "custom-dataset",
    moduleId: "data-pipeline",
    title: "Define how one sample is retrieved",
    subtitle: "A Dataset maps an index to a paired example.",
    source: [26, 27],
    kind: "loader",
    diagramTitle: "An index retrieves one paired sample",
    diagramDescription:
      "A map-style Dataset receives an integer index and returns that row's features and matching target; TensorDataset is a built-in option for aligned tensors.",
    objective:
      "Implement __init__, __len__ and __getitem__, and choose between a custom Dataset and TensorDataset.",
    intuition:
      "A Dataset is a catalogue with a rule for looking up one record. It answers how many records exist and what the feature/label pair at an index contains.",
    equation: "dataset[i] → (features[i], label[i])",
    numerical:
      "If there are 12 rows, len(dataset) is 12; asking for index 3 returns feature shape [30] and a paired label shape [1].",
    shapes: [
      ["Stored features", "[N,30]", "All tabular rows."],
      ["Stored targets", "[N,1]", "One float binary label per row."],
      ["One item", "[30] + [1]", "A feature vector and its corresponding target."],
    ],
    sections: [
      [
        "custom",
        "Implement the map-style contract",
        "A custom Dataset generally stores or references data in __init__, returns the sample count from __len__, and retrieves a paired item in __getitem__(index). Transformations that need to happen per sample can live in __getitem__.\nThe DataLoader calls these methods as it requests indices; training code usually loops over the loader rather than calling magic methods directly.",
      ],
      [
        "built-in",
        "Use TensorDataset when tensors are already aligned",
        "TensorDataset pairs tensors by their shared first dimension, which is concise for in-memory feature and target arrays. A custom Dataset is useful for file-backed records, transforms, variable-length examples, or more involved lookup.\nFor classification, keep features and labels paired under the same index. Convert features to float32; encode text labels explicitly before they reach a numeric loss.",
      ],
    ],
    snippet: "custom-dataset",
    template: "tensor-dataset",
    additional: ["dataset-vs-dataloader"],
    pitfalls: [
      "Feature and label lengths must match.",
      "Do not sort or filter features independently of their labels.",
      "String labels need an explicit mapping before tensor construction.",
      "Dataset handles sample access; it does not by itself create batches or shuffle indices.",
    ],
    production:
      "Avoid loading a huge dataset into RAM just to wrap it in TensorDataset. A custom Dataset can lazily decode files and apply transforms per sample.",
    revision: [
      "__init__ stores or configures sample access.",
      "__len__ reports sample count.",
      "__getitem__ returns one paired sample.",
      "TensorDataset is convenient for aligned tensors already in memory.",
    ],
    questions: [
      [
        "What does __getitem__(i) return?",
        "The feature and target belonging to index i.",
      ],
      [
        "When is TensorDataset enough?",
        "When aligned tensors already exist and sample lookup needs no custom logic.",
      ],
      [
        "Which dimension must match across tensors in TensorDataset?",
        "Their leading sample dimension.",
      ],
      [
        "Where does DataLoader use len and getitem?",
        "It uses dataset length for sampling/batch counts and getitem to fetch each selected sample.",
      ],
    ],
  }),
  lesson({
    id: "dataloader-pipeline",
    moduleId: "data-pipeline",
    title: "Follow indices into minibatches",
    subtitle: "Sampler, dataset lookup and collation form one pipeline.",
    source: [27, 29],
    kind: "loader",
    diagramTitle: "Index lookup becomes a model batch",
    diagramDescription:
      "A sampler chooses dataset indices, Dataset retrieves paired rows, DataLoader applies collate_fn, and the model receives a batch tensor.",
    objective:
      "Explain sampler, DataLoader, batch_size, shuffle, default collation, RandomSampler and SequentialSampler.",
    intuition:
      "The sampler writes a pickup list, the Dataset retrieves each indexed package, and collation packs the selected records into a batch the model can carry.",
    equation: "indices → dataset[index] → collate_fn(samples) → batch",
    numerical:
      "Ten rows and batch_size=4 with drop_last=False yield batches of sizes 4, 4 and 2. A partial final batch is valid.",
    shapes: [
      ["Index list", "[B] integers", "Sample IDs chosen for this batch."],
      ["Fetched samples", "B × ([30]+[1])", "Paired feature and target items."],
      ["Collated batch", "[B,30] + [B,1]", "Dense tensors supplied to the model."],
    ],
    sections: [
      [
        "sampler",
        "Choose sample order without breaking pairs",
        "shuffle=True makes DataLoader use random sampling for a map-style dataset; shuffle=False uses sequential sampling. You may pass RandomSampler or SequentialSampler explicitly, but do not set shuffle and sampler together.\nA custom sampler can address imbalance or specialized order. A sampler chooses indices; exact per-batch class quotas require a batch sampler or a carefully designed sampling scheme, not a vague promise from RandomSampler.",
      ],
      [
        "batch",
        "DataLoader fetches and combines samples",
        "DataLoader divides the selected index sequence into chunks of batch_size, asks the Dataset for each item, and applies collate_fn. Default collation stacks same-shaped tensors while keeping feature-label pairs together.\nA final short batch is normal when drop_last=False. The existing ten-sample slider shows how batch sizes 1–5 change batch count and dropped records. Supplemental: IterableDataset is useful for streams that do not have stable random-access indices; each worker must shard its stream deliberately, and reported length/batch counts can be approximate.",
      ],
    ],
    snippet: "dataloader-pipeline",
    template: "loader-options",
    pitfalls: [
      "Do not pass both sampler and shuffle to the same map-style DataLoader.",
      "Custom samplers must preserve the intended sample/label association.",
      "drop_last=True discards data for that epoch.",
      "len(loader) can be sampler- or dataset-dependent; it is not always ceil(N/B).",
    ],
    production:
      "Training often shuffles; evaluation usually preserves order and includes every example. Keep a deterministic evaluation loader so predictions remain aligned when collecting per-row outputs.",
    revision: [
      "Sampler chooses indices.",
      "Dataset fetches one sample for each index.",
      "collate_fn combines samples into a batch.",
      "DataLoader coordinates the process and applies batch options.",
    ],
    questions: [
      [
        "What is the role of a sampler?",
        "It chooses the dataset indices to fetch and in what order.",
      ],
      [
        "Which function stacks the fetched samples by default?",
        "The DataLoader's default collate function.",
      ],
      ["How many batches for N=10, B=4, drop_last=False?", "Three: 4, 4 and 2 samples."],
      [
        "What does shuffle=True select by default?",
        "Random sampling for a map-style dataset.",
      ],
    ],
  }),
  lesson({
    id: "collation-workers",
    moduleId: "data-pipeline",
    title: "Pad variable data and tune loading safely",
    subtitle:
      "Custom collation handles ragged items; worker settings depend on the environment.",
    source: [28, 28],
    kind: "collate",
    diagramTitle: "Ragged sequences become a padded batch",
    diagramDescription:
      "Three token sequences of lengths two, four and three are padded with zeros to a [3,4] tensor, with original lengths retained.",
    objective:
      "Write a custom collate_fn for variable-length examples and choose worker, pin-memory and drop-last settings deliberately.",
    intuition:
      "Collation is the packing table. Equal-sized parts stack directly; ragged text needs padding and a length label before it fits a rectangular batch.",
    equation: "[2], [4], [3] → pad_sequence → [3,4]",
    numerical:
      "Token lengths 2, 4 and 3 become padded rows [4,5,0,0], [7,8,9,10], and [2,3,4,0].",
    shapes: [
      ["Unpadded items", "[2], [4], [3]", "Variable-length token sequences."],
      ["Padded tokens", "[3,4] long", "Batch-first integer token matrix."],
      ["Lengths/labels", "[3] long each", "Original lengths and encoded class IDs."],
    ],
    sections: [
      [
        "collate",
        "Give a batch its own assembly rule",
        "When sample shapes differ, default stacking cannot make a dense rectangular tensor. A custom collate_fn can pad sequences, preserve lengths, create masks, or encode metadata while maintaining order.\nThe example uses pad_sequence(batch_first=True), stores original lengths and explicitly maps string labels to integer IDs. Integer token IDs and class-index labels use torch.long; a binary BCE target instead uses floating point.",
      ],
      [
        "workers",
        "Start portable and measure",
        "num_workers controls worker processes for map-style loading. Start at zero, especially for a notebook or a small in-memory dataset. On Windows, multiprocessing uses spawn: put loader creation and iteration behind an if __name__ == '__main__' guard in a standalone script.\npin_memory is useful when staging batches for supported accelerators; it is not a universal speed switch. It is enabled conditionally in the example, and non_blocking device transfer is only meaningful when an appropriate pinned-memory/device path exists.",
      ],
    ],
    snippet: "text-collate",
    template: "windows-workers",
    pitfalls: [
      "Default collation cannot stack differently sized sequences without padding.",
      "Padding token value and attention mask semantics must match the model.",
      "More workers can slow small in-memory datasets or duplicate state under spawn.",
      "pin_memory does not move a tensor to the GPU; it affects host-memory staging.",
      "drop_last is a batch choice, not a fix for every shape error.",
    ],
    production:
      "Benchmark worker counts with the actual storage and transforms. Keep Windows worker functions top-level and guard process creation; leave num_workers=0 as the portable baseline.",
    revision: [
      "collate_fn combines sample objects into a batch.",
      "pad_sequence can turn ragged token rows into a rectangular tensor.",
      "num_workers=0 is the simplest portable starting point.",
      "Pin memory can help host-to-accelerator transfer when used conditionally.",
    ],
    questions: [
      [
        "Why write a custom collate_fn?",
        "To combine examples that need padding, masking or other per-batch assembly.",
      ],
      [
        "What shape results from lengths 2,4,3 with batch_first=True?",
        "[3,4], plus a length vector [3].",
      ],
      [
        "What is the safest starting worker count on Windows?",
        "num_workers=0; use a main guard before enabling multiprocessing in scripts.",
      ],
      [
        "Does pin_memory put data on the GPU?",
        "No. It pins host memory so eligible host-to-device transfers can be more efficient.",
      ],
    ],
  }),
  lesson({
    id: "idiomatic-classifier",
    moduleId: "training",
    title: "Train the idiomatic full-batch classifier",
    subtitle: "Replace manual math while preserving the source's 30-feature task.",
    source: [21, 24],
    kind: "module",
    diagramTitle: "A registered model learns from paired rows",
    diagramDescription:
      "Thirty float features flow through a registered Linear layer to one raw logit, compared with a paired float binary target by BCEWithLogitsLoss.",
    objective:
      "Assemble a stable CPU binary classifier with a model class, BCEWithLogitsLoss, SGD and matching float32 tensors.",
    intuition:
      "The dataset is the same familiar set of paired feature rows; modules, losses and optimizers take over the reusable bookkeeping around those rows.",
    equation: "X [N,30] → Linear → logits [N,1] → BCEWithLogitsLoss(y [N,1])",
    numerical:
      "For N=128, the model emits 128 logits. Thresholding logits at zero is equivalent to thresholding sigmoid probabilities at 0.5.",
    shapes: [
      ["Features", "[128,30] float32", "Synthetic Wisconsin-style feature rows."],
      ["Logits", "[128,1] float32", "Unnormalized model scores."],
      ["Targets", "[128,1] float32", "B=0 and M=1 binary target convention."],
    ],
    sections: [
      [
        "data",
        "Use a deterministic synthetic fallback",
        "The source code contains a clipped remote CSV address and later notes that it returned a 404. This lesson does not embed or call a URL. It generates small synthetic 30-feature rows so the pipeline runs offline.\nThe earlier Phase 03 optional local CSV reader remains available for a user-supplied file. Its B=0/M=1 mapping and train-only scaling discipline still apply when preparing a real dataset.",
      ],
      [
        "model",
        "Train on raw logits and inspect progress",
        "A registered Linear(30,1) model returns logits; BCEWithLogitsLoss consumes them alongside targets of the same [N,1] shape. The optimizer clears gradients, backward computes them and step updates the model.\nThe snippet uses a full batch to isolate the idiomatic model replacement. The subsequent DataLoader lesson adds minibatches, separate validation/test loaders, and evaluation with model.eval() and inference_mode().",
      ],
    ],
    snippet: "idiomatic-classifier",
    template: "state-dict-modes",
    pitfalls: [
      "Do not rely on a clipped or broken source URL.",
      "Targets and logits must have identical shapes for binary loss.",
      "Keep model parameters, features and targets on the same device.",
      "Use BCEWithLogitsLoss on logits; apply sigmoid only when probabilities are needed for display or metrics.",
    ],
    production:
      "Fit scaling only on training rows, use validation data for model choices, and reserve a test split for final reporting. The synthetic example teaches the pipeline; it is not a clinical model.",
    revision: [
      "nn.Linear(30,1) emits logits [B,1].",
      "BCEWithLogitsLoss consumes logits and float targets.",
      "SGD owns registered parameters from model.parameters().",
      "Synthetic data is a runnable fallback, not evidence of clinical performance.",
    ],
    questions: [
      [
        "What does the source dataset URL do in the reader?",
        "Nothing: it is clipped/broken in the export and the runnable example uses synthetic data instead.",
      ],
      ["What does nn.Linear(30,1) return?", "One raw logit per row, shape [B,1]."],
      [
        "Why does BCEWithLogitsLoss pair naturally with this model?",
        "It accepts raw logits and is numerically stable without a separate sigmoid in the loss path.",
      ],
      [
        "What does a zero logit mean for a 0.5 probability threshold?",
        "sigmoid(0)=0.5, so classify positive when logit is at least zero.",
      ],
    ],
  }),
  lesson({
    id: "batched-classifier",
    moduleId: "data-pipeline",
    title: "Train and evaluate with DataLoader",
    subtitle: "Use train, validation and test batches without leaking preprocessing.",
    source: [29, 40],
    kind: "pipeline",
    diagramTitle: "Thirty features travel from split to batch to model",
    diagramDescription:
      "A train-only scaler transforms separate splits; train rows are shuffled into batches, while validation and test are iterated in stable order and evaluated without gradients.",
    objective:
      "Combine paired datasets and loaders with a train/eval loop, leakage-free scaling, validation/test metrics and state_dict checkpoints.",
    intuition:
      "Three labeled trays keep decisions honest: training updates weights, validation helps select a model, and the final test tray is opened once for reporting.",
    equation:
      "fit scaler on train → transform all splits → train batches → validate → final test",
    numerical:
      "For 120 training rows and batch_size=16, drop_last=False yields eight batches: seven of size 16 and one of size 8. Validation/test rows never contribute to the scaler mean or scale.",
    shapes: [
      ["Split", "train/val/test [N,30]", "Disjoint records with paired labels."],
      ["Mini-batch", "[B,30] + [B,1]", "Current features and binary labels."],
      [
        "Prediction",
        "logits [B,1]",
        "Threshold at zero for the 0.5 probability decision.",
      ],
    ],
    sections: [
      [
        "leakage",
        "Prepare splits before fitting learned preprocessing",
        "The runnable synthetic pipeline creates separate train, validation and test indices. It calculates mean and scale only from training features, then applies those saved values to each split.\nDo not fit StandardScaler on all rows before splitting. The validation split is for choices such as epochs or thresholds; the test split remains held out until the final evaluation.",
      ],
      [
        "loop",
        "Train in batches and evaluate in eval mode",
        "The training loader shuffles paired samples. Each batch runs zero_grad(set_to_none=True), forward, loss, backward and step. Validation and test loaders use stable ordering and no gradient tracking; report accuracy over all samples rather than averaging batch accuracies when batch sizes differ.\nSave model.state_dict(), not an arbitrary whole-model pickle. Load with map_location and weights_only=True when supported, then call eval before inference. Never load a checkpoint from an untrusted source.",
      ],
    ],
    snippet: "batched-classifier",
    template: "state-dict-modes",
    pitfalls: [
      "Do not fit preprocessing on validation or test rows.",
      "Do not select hyperparameters repeatedly against the test set.",
      "Do not average batch accuracies equally when batch sizes differ; aggregate counts.",
      "Evaluation needs eval() and a gradient-disabled context.",
      "Save state_dict weights and load safely; avoid untrusted full-object pickle files.",
    ],
    production:
      "Use the same preprocessing state at inference that was fit on train data. Save it with the model artifact. For production checkpoints, also record label mapping, feature order, package versions and evaluation context.",
    revision: [
      "DataLoader yields paired minibatches.",
      "Fit preprocessing on training rows only.",
      "Call train() before updates and eval() for validation/test.",
      "state_dict stores weights; load safely and explicitly.",
    ],
    questions: [
      [
        "Why fit the scaler only on training rows?",
        "Using held-out distribution statistics leaks information into training and evaluation.",
      ],
      [
        "What changes between train and evaluation loops?",
        "Training computes gradients and steps an optimizer; evaluation calls eval() under inference_mode and performs no updates.",
      ],
      [
        "How many batches for 120 rows with B=16 and drop_last=False?",
        "Eight batches: seven of 16 and one of 8.",
      ],
      [
        "What should a basic model checkpoint contain?",
        "A state_dict of weights, loaded with an explicit map_location and safe weights-only loading where available.",
      ],
    ],
  }),
];
