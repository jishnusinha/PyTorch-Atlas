import type { Lesson, Snippet } from "./types";

const make = (
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
  lines: code
    .split("\n")
    .map(
      (line, i) =>
        `Line ${i + 1} · ${line.trim().startsWith("import ") ? "Import the dependencies for this example." : line.trim().startsWith("assert ") ? "Check the stated shape, dtype or mathematical result." : line.trim().includes("backward") ? "Propagate the loss derivative through the tracked operations." : line.trim().startsWith("print(") ? "Display values to compare with the expected output." : line.trim().startsWith("with torch.no_grad") ? "Apply the manual update without recording it in autograd." : `Follow this operation in the ${shapes} tensor flow.`}`,
    ),
});
const s: Snippet[] = [
  make(
    "manual-preprocessing",
    "Split and scale synthetic classification data",
    "Create a deterministic 30-feature dataset, stratify labels and fit scaling on training rows only.",
    "X [240,30] → train/validation/test [N,30]; labels [N,1]",
    `import torch

def make_data(n=240, features=30):
    g=torch.Generator().manual_seed(7)
    x=torch.randn((n,features),generator=g,dtype=torch.float32)
    score=1.5*x[:,0]-1.2*x[:,1]+0.8*x[:,2]
    y=(score>0).to(torch.float32).unsqueeze(1)
    return x,y

x,y=make_data()
g=torch.Generator().manual_seed(11)
train=[]; validation=[]; test=[]
for label in (0.,1.):
    ids=(y[:,0]==label).nonzero().flatten()
    ids=ids[torch.randperm(len(ids),generator=g)]
    a=int(.70*len(ids)); b=int(.85*len(ids))
    train.append(ids[:a]); validation.append(ids[a:b]); test.append(ids[b:])
train_ids=torch.cat(train); val_ids=torch.cat(validation); test_ids=torch.cat(test)
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
assert x_train.shape[1]==30 and y_train.shape==(len(train_ids),1)
assert x_train.dtype==torch.float32 and len(set(train_ids.tolist()).intersection(test_ids.tolist()))==0
print(f"train={tuple(x_train.shape)}, val={tuple(x_val.shape)}, test={tuple(x_test.shape)}")`,
    "train/validation/test each [N,30]; float32; disjoint stratified rows",
  ),
  make(
    "local-csv-loader",
    "Read an optional local Wisconsin CSV",
    "Use a local file when supplied; never download data or assume a truncated URL.",
    "CSV features [N,30] → X float32 [N,30], y float32 [N,1]",
    `import csv
from pathlib import Path
import torch

def read_local_csv(path):
    path=Path(path)
    if not path.is_file():
        return None
    with path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    if not rows or "diagnosis" not in rows[0]:
        raise ValueError("CSV needs a diagnosis column")
    ignored={"id","unnamed: 32","diagnosis"}
    columns=[c for c in rows[0] if c.lower() not in ignored]
    if len(columns)!=30:
        raise ValueError(f"expected 30 feature columns, found {len(columns)}")
    mapping={"B":0.0,"M":1.0}
    labels=[r["diagnosis"].strip().upper() for r in rows]
    if any(label not in mapping for label in labels):
        raise ValueError("diagnosis must be B or M")
    x=torch.tensor([[float(r[c]) for c in columns] for r in rows],dtype=torch.float32)
    y=torch.tensor([mapping[label] for label in labels],dtype=torch.float32).reshape(-1,1)
    assert x.shape==(len(rows),30) and y.shape==(len(rows),1)
    return x,y

dataset=read_local_csv("data.csv")
if dataset is None:
    print("No local data.csv; run the synthetic preprocessing example instead.")
else:
    print("Loaded local rows:",tuple(dataset[0].shape),"labels:",tuple(dataset[1].shape))`,
    "With local file: X [N,30], y [N,1]; otherwise an explicit no-file message",
  ),
  make(
    "manual-forward",
    "Build MySimpleNN and trace its forward pass",
    "Store a trainable [30,1] matrix and scalar bias; return logits and probabilities.",
    "X [B,30] @ W [30,1] + b [1] → logits/probabilities [B,1]",
    `import torch

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
print("weights",tuple(model.weights.shape),"logits",tuple(logits.shape))`,
    "weights (30,1); logits (4,1); probabilities in [0,1]",
  ),
  make(
    "manual-bce",
    "Derive binary cross-entropy from probabilities",
    "Align target shapes, clip probabilities and compare with the stable logits loss.",
    "logits/probabilities/targets [3,1] → scalar loss []",
    `import torch
import torch.nn.functional as F

logits=torch.tensor([[-2.0],[0.0],[2.0]],dtype=torch.float32)
targets=torch.tensor([[0.0],[1.0],[1.0]],dtype=torch.float32)
probabilities=torch.sigmoid(logits)
epsilon=1e-7
p=probabilities.clamp(epsilon,1-epsilon)
manual=-(targets*torch.log(p)+(1-targets)*torch.log(1-p)).mean()
stable=F.binary_cross_entropy_with_logits(logits,targets)
assert manual.ndim==0 and torch.allclose(manual,stable,atol=1e-6)
print(f"manual={manual.item():.6f}, logits_loss={stable.item():.6f}")`,
    "manual and BCEWithLogitsLoss agree to float32 tolerance",
  ),
  make(
    "manual-one-step",
    "Run one manual forward/backward/update step",
    "Compute a scalar BCE loss, backpropagate, update leaves under no_grad and clear gradients.",
    "X [8,30] → logits/probabilities [8,1] → loss []; grads W [30,1], b [1]",
    `import torch
import torch.nn.functional as F

torch.manual_seed(3)
x=torch.randn((8,30),dtype=torch.float32)
y=(x[:,:1]>0).to(torch.float32)
w=torch.zeros((30,1),requires_grad=True); b=torch.zeros((1,),requires_grad=True)
logits=x@w+b; loss=F.binary_cross_entropy_with_logits(logits,y)
before=loss.item(); loss.backward()
assert w.grad.shape==w.shape and b.grad.shape==b.shape
with torch.no_grad():
    w-=0.1*w.grad
    b-=0.1*b.grad
w.grad.zero_(); b.grad.zero_()
after=F.binary_cross_entropy_with_logits(x@w+b,y)
assert after.item()<before and w.grad.count_nonzero()==0
print(f"loss {before:.4f} → {after.item():.4f}; gradients cleared")`,
    "one update lowers toy loss; gradients are zero before the next pass",
  ),
  make(
    "gradient-accumulation-demo",
    "Why the next pass must clear gradients",
    "Run two mini-iterations and inspect the derivative before and after zeroing.",
    "w [] → loss₁ [] → grad []; loss₂ [] → accumulated grad []",
    `import torch

w=torch.tensor(2.0,dtype=torch.float32,requires_grad=True)
(w**2).backward()
first=w.grad.item()
(3*w).backward()
accumulated=w.grad.item()
assert first==4.0 and accumulated==7.0
w.grad.zero_()
(3*w).backward()
fresh=w.grad.item()
assert fresh==3.0
print(f"iteration 1={first:.0f}; accumulated={accumulated:.0f}; cleared iteration 2={fresh:.0f}")`,
    "iteration 1=4; accumulated=7; cleared iteration 2=3",
  ),
  make(
    "full-manual-classifier",
    "Train a complete CPU-only manual classifier",
    "Reconstruct preprocessing, [30,1] parameters, BCE, backward, SGD and held-out evaluation without external data.",
    "train X [N,30], W [30,1] → probability [N,1]; test predictions [N,1]",
    `import torch

def dataset(n=240):
    g=torch.Generator().manual_seed(7)
    x=torch.randn((n,30),generator=g,dtype=torch.float32)
    score=1.5*x[:,0]-1.2*x[:,1]+0.8*x[:,2]
    return x,(score>0).to(torch.float32).unsqueeze(1)
def bce(prob,y):
    p=prob.clamp(1e-7,1-1e-7)
    return -(y*torch.log(p)+(1-y)*torch.log(1-p)).mean()
class MySimpleNN:
    def __init__(self):
        self.weights=(torch.randn((30,1))*0.05).requires_grad_()
        self.bias=torch.zeros((1,),requires_grad=True)
    def forward(self,x):
        logits=x@self.weights+self.bias
        return logits,torch.sigmoid(logits)
    def __call__(self,x):
        return self.forward(x)

torch.manual_seed(13)
x,y=dataset()
g=torch.Generator().manual_seed(11)
sets=[[],[],[]]
for label in (0.,1.):
    ids=(y[:,0]==label).nonzero().flatten()
    ids=ids[torch.randperm(len(ids),generator=g)]
    a=int(.70*len(ids)); b=int(.85*len(ids))
    for bucket,part in zip(sets,(ids[:a],ids[a:b],ids[b:])):
        bucket.append(part)
train_ids,val_ids,test_ids=[torch.cat(parts) for parts in sets]
x_train,x_val,x_test=x[train_ids],x[val_ids],x[test_ids]
y_train,y_val,y_test=y[train_ids],y[val_ids],y[test_ids]
mean=x_train.mean(0,keepdim=True)
scale=x_train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)
x_train=(x_train-mean)/scale
x_val=(x_val-mean)/scale
x_test=(x_test-mean)/scale
model=MySimpleNN()
initial=bce(model(x_train)[1],y_train).item()
learning_rate=0.12
for epoch in range(100):
    logits,probabilities=model(x_train)
    loss=bce(probabilities,y_train)
    loss.backward()
    with torch.no_grad():
        model.weights-=learning_rate*model.weights.grad
        model.bias-=learning_rate*model.bias.grad
    model.weights.grad.zero_(); model.bias.grad.zero_()
final=bce(model(x_train)[1],y_train)
with torch.inference_mode():
    test_logits,test_probabilities=model(x_test)
    predicted=(test_probabilities>=0.5).to(torch.float32)
    accuracy=(predicted==y_test).to(torch.float32).mean()
assert model.weights.shape==(30,1) and model.bias.shape==(1,)
assert x_train.dtype==y_train.dtype==torch.float32
assert test_logits.shape==test_probabilities.shape==y_test.shape
assert final.item()<initial and accuracy.item()>.75
print(f"train loss {initial:.4f} → {final.item():.4f}; test accuracy={accuracy.item():.3f}; test shape={tuple(predicted.shape)}")`,
    "Training loss decreases; output and target both [N,1], CPU float32",
  ),
  make(
    "binary-evaluation",
    "Threshold probabilities and read a confusion matrix",
    "Use a stable [N,1] shape and make the positive-label direction explicit.",
    "probability/target/prediction [N,1] → confusion counts scalars",
    `import torch

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
print(f"threshold=.5 · TP={tp.item()} TN={tn.item()} FP={fp.item()} FN={fn.item()} · accuracy={accuracy.item():.2f}")`,
    "TP=1 TN=1 FP=1 FN=1; accuracy=0.50",
  ),
];
const sections = (a: string, ap: string[], b: string, bp: string[]) => [
  { id: "understand", title: a, paragraphs: ap },
  { id: "mechanics", title: b, paragraphs: bp },
];
const qa = (question: string, answer: string) => ({ question, answer });
const lessons: Lesson[] = [
  {
    id: "binary-classification",
    moduleId: "training",
    title: "A binary classifier is a learning workflow.",
    subtitle:
      "Frame the Wisconsin breast-cancer example and verify what each label means.",
    minutes: 9,
    source: {
      start: 6,
      end: 7,
      kind: "source-derived",
      note: "The source presents this as a pipeline demonstration, not a guaranteed benchmark.",
    },
    objectives: [
      "Identify one row, 30 features and one binary diagnosis.",
      "State the B/M-to-0/1 mapping explicitly.",
      "Separate the educational goal from a performance claim.",
    ],
    intuition:
      "Each case is a row in a table; the model combines 30 measurements into one score, then maps that score to a probability for the positive class.",
    equation: "x ∈ ℝ³⁰ → z=xW+b ∈ ℝ → p=sigmoid(z) ∈ (0,1)",
    numerical:
      "A single case has 30 measurements. The PDF's LabelEncoder maps alphabetical B→0 and M→1; state this mapping rather than assuming class 1 is obvious. The output is P(M) only under that convention.",
    shapes: [
      {
        name: "One feature row",
        shape: "[1,30]",
        meaning: "One case, thirty numeric measurements.",
      },
      {
        name: "Diagnosis",
        shape: "[1,1]",
        meaning: "One binary target; B=0, M=1 in this mapping.",
      },
      {
        name: "Model output",
        shape: "[1,1]",
        meaning: "Estimated positive-class probability.",
      },
    ],
    diagram: {
      kind: "pipeline",
      title: "One row becomes one probability",
      description: "Select a pipeline stage to inspect its role, shape and equation.",
    },
    sections: sections(
      "The notebook is a teaching example",
      [
        "The source uses the Wisconsin breast-cancer diagnosis table and a single affine score followed by sigmoid. Its aim is to expose the training steps before introducing PyTorch modules and optimizers.",
        "A high test score on one split is not a benchmark guarantee. It depends on split, preprocessing, initialization and training choices; later chapters use controlled validation.",
      ],
      "Make labels and the prediction target explicit",
      [
        "The source drops the id and empty Unnamed: 32 columns, keeps 30 numeric features, and encodes diagnosis labels. Alphabetical encoding assigns B=0, M=1, so sigmoid output is interpreted as the probability of M.",
        "Check class counts and labels before training. If the dataset or label mapping changes, the meaning of the 0.5 threshold changes too.",
      ],
    ),
    snippet: s[1],
    template: s[0],
    pitfalls: [
      "Do not treat an ID column as a predictive measurement.",
      "Do not say class 1 means malignant unless your encoding confirms it.",
      "Do not report source-demo performance as a guaranteed result.",
      "A single train/test split is not model selection or a validation protocol.",
    ],
    production:
      "For real use, document label direction, stratify the split, retain a validation set for choices and reserve test data for final reporting.",
    revision: [
      "One row has 30 feature values.",
      "B=0 and M=1 must be stated and checked.",
      "Sigmoid produces positive-class probability.",
      "The source's goal is pipeline mechanics, not a benchmark.",
    ],
    questions: [
      qa(
        "What does sigmoid output mean here?",
        "With the stated B=0, M=1 encoding, it estimates P(M).",
      ),
      qa(
        "Why remove id?",
        "It identifies a row but is not one of the 30 measured features.",
      ),
      qa(
        "Why not promise the PDF's test accuracy?",
        "A single demo result depends on its split and training choices and is not a guaranteed benchmark.",
      ),
      qa(
        "What does a 0.5 threshold decide?",
        "It predicts label 1 when the estimated probability is at least 0.5.",
      ),
    ],
  },
  {
    id: "preprocessing",
    moduleId: "training",
    title: "Fit preprocessing on training rows only.",
    subtitle:
      "Split, encode, standardize and convert without leaking held-out information.",
    minutes: 12,
    source: {
      start: 7,
      end: 7,
      kind: "source-derived",
      note: "A stratified validation split, synthetic fallback and local-file loader are supplemental repairs.",
    },
    objectives: [
      "Keep features and diagnosis aligned through a split.",
      "Fit scaling statistics on training data only.",
      "Create float32 tensors with explicit target shapes.",
    ],
    intuition:
      "Learn the ruler from the training examples, then use that same ruler to measure validation and test rows. If held-out rows help set the ruler, information has leaked.",
    equation: "μ_train=mean(X_train); σ_train=std(X_train); X_scaled=(X−μ_train)/σ_train",
    numerical:
      "The source splits before scaling. Fit the scaler only on X_train; apply its mean and scale unchanged to validation/test. StandardScaler uses population standard deviation (ddof=0). Convert features and targets to float32, with targets reshaped [N,1].",
    shapes: [
      { name: "Raw features", shape: "[N,30]", meaning: "Rows and 30 measured values." },
      {
        name: "Train / validation / test",
        shape: "[N₁,30] / [N₂,30] / [N₃,30]",
        meaning: "Disjoint, stratified row groups.",
      },
      {
        name: "Train-only scaler",
        shape: "mean [1,30], scale [1,30]",
        meaning: "Fit only from training rows.",
      },
      { name: "Targets", shape: "[N,1]", meaning: "float32 values 0 or 1." },
    ],
    diagram: {
      kind: "pipeline",
      title: "Split first; fit the ruler on train only",
      description:
        "The preprocessing stage makes the train-only statistics and held-out transform visible.",
    },
    sections: sections(
      "Reconstruct rather than trust the clipped export",
      [
        "The source code line for a GitHub URL is cut off, so it is not a runnable data source. The PDF also shows Pandas, train_test_split, StandardScaler and LabelEncoder; we explain that workflow without embedding a URL.",
        "The examples run on deterministic synthetic data by default. A separate standard-library CSV reader accepts a user-provided local file and reports a clear message when it is absent.",
      ],
      "Prevent leakage and shape mismatches",
      [
        "Split rows before fitting any learned preprocessing. Fit mean/scale on training rows, then transform validation/test with the same values. The examples stratify by the binary label and keep row indices disjoint.",
        "Map B→0 and M→1 explicitly. Use float32 for features and targets, and reshape targets to [N,1] to match this single-output model. Validation guides choices; test is held until final evaluation.",
      ],
    ),
    snippet: s[0],
    template: s[1],
    pitfalls: [
      "Fitting a scaler before splitting leaks held-out distribution information.",
      "Never use fit_transform on validation or test rows.",
      "Map diagnosis text explicitly and verify unique encoded values.",
      "Keep both logits/probabilities and targets at [N,1], not [N,1] versus [N].",
    ],
    production:
      "Persist the training split and its fitted preprocessing values with the model. For cross-validation, fit a new scaler independently inside each training fold.",
    revision: [
      "Split before fitting learned preprocessing.",
      "Fit mean and standard deviation on training rows only.",
      "Apply the same transform to validation and test.",
      "Use explicit label mapping and matching float32 target shapes.",
    ],
    questions: [
      qa(
        "Why split before standardization?",
        "So held-out values do not influence training means or scales.",
      ),
      qa(
        "What does the source's alphabetical label encoding mean?",
        "B maps to 0 and M to 1; therefore sigmoid is interpreted as P(M).",
      ),
      qa(
        "Why use target shape [N,1]?",
        "It matches the one-output model and prevents accidental broadcasting against [N].",
      ),
      qa(
        "What is the synthetic fallback for?",
        "A reproducible, download-free exercise of the full workflow; it is not the Wisconsin data.",
      ),
    ],
  },
  {
    id: "manual-forward",
    moduleId: "training",
    title: "A manual neuron maps 30 features to one score.",
    subtitle: "Store W and b as gradient-tracked leaves and trace the batch dimensions.",
    minutes: 10,
    source: {
      start: 7,
      end: 8,
      kind: "source-derived",
      note: "The source uses float64 and a clipped matrix declaration; this reconstruction uses explicit float32 and small seeded initialization.",
    },
    objectives: [
      "Create trainable weights [30,1] and bias [1].",
      "Calculate X@W+b and sigmoid probabilities.",
      "Keep batch outputs [B,1].",
    ],
    intuition:
      "Each feature contributes its value times one learned weight. The bias shifts the score for all examples; sigmoid turns the score into a number between zero and one.",
    equation: "Z = XW + b      P = σ(Z) = 1/(1+e⁻ᶻ)",
    numerical:
      "For batch size B=4: X is [4,30], W is [30,1], so XW and Z are [4,1]. Broadcasting b[1] across the four rows keeps [4,1]; sigmoid preserves that shape.",
    shapes: [
      {
        name: "Features",
        shape: "[B,30]",
        meaning: "B cases and 30 standardized columns.",
      },
      {
        name: "Weights",
        shape: "[30,1]",
        meaning: "One trainable coefficient per feature.",
      },
      {
        name: "Bias",
        shape: "[1]",
        meaning: "One trainable offset broadcast across rows.",
      },
      {
        name: "Logits / probability",
        shape: "[B,1]",
        meaning: "One score and one sigmoid output per case.",
      },
    ],
    diagram: {
      kind: "pipeline",
      title: "Feature row → linear score → probability",
      description:
        "Choose the logits or probability stage to inspect shape, equation and expected range.",
    },
    sections: sections(
      "Two leaves define the model",
      [
        "MySimpleNN stores weights and bias as floating-point leaf tensors with requires_grad=True. A modest seeded initialization avoids the very large initial sigmoid saturation that an unscaled rand initialization can cause.",
        "This plain class is intentionally manual: it is not an nn.Module. Phase 04 will introduce registered parameters and module behavior.",
      ],
      "Use the callable model boundary",
      [
        "The forward method computes logits with matrix multiplication and adds the bias. __call__ delegates to forward so the examples use model(x), the standard model call pattern.",
        "Keep logits and probabilities as [B,1]. Squeezing every dimension can erase the batch axis when B=1. The thresholded prediction should also keep a two-dimensional single-output shape.",
      ],
    ),
    snippet: s[2],
    template: s[2],
    pitfalls: [
      "Weights shaped [1,30] do not match X @ W; use [30,1].",
      "Integer tensors cannot require gradients; initialize float32 leaves.",
      "Calling forward directly bypasses the usual call boundary; use model(x).",
      "An unqualified squeeze() can remove the batch axis for one example.",
    ],
    production:
      "For real datasets, initialize weights with a scale appropriate to the input features. The next phase replaces this plain container with nn.Module and nn.Linear.",
    revision: [
      "X [B,30] times W [30,1] yields [B,1].",
      "Bias [1] broadcasts across batch rows.",
      "Sigmoid maps logits to probabilities.",
      "Keep the target and prediction shapes equal.",
    ],
    questions: [
      qa("What is the weight shape for 30 inputs and one output?", "[30,1]."),
      qa("What shape results from X[8,30] @ W[30,1]?", "[8,1]."),
      qa(
        "Does the bias change the output rank?",
        "No. Its one value broadcasts across rows, preserving [B,1].",
      ),
      qa(
        "Why call model(x)?",
        "It gives a consistent callable model interface; nn.Module also runs hooks and other call behavior.",
      ),
    ],
  },
  {
    id: "manual-loss",
    moduleId: "training",
    title: "Binary cross-entropy scores probability errors.",
    subtitle:
      "Derive the clipped probability formula and compare its logits-safe counterpart.",
    minutes: 11,
    source: {
      start: 8,
      end: 9,
      kind: "source-derived",
      note: "Clipping safeguards the handwritten probability formula; BCEWithLogitsLoss is a recommended numerical-stability comparison.",
    },
    objectives: [
      "Write binary cross-entropy for a 0/1 target.",
      "Explain why probability clipping is used.",
      "Compare probability BCE with BCEWithLogitsLoss.",
    ],
    intuition:
      "A confident correct probability receives a small penalty; a confident wrong probability receives a large one. The logarithm makes confident mistakes especially costly.",
    equation: "BCE(y,p)=−[y log(p)+(1−y)log(1−p)]",
    numerical:
      "For y=1,p=.8 the loss is −log(.8)≈.223. For y=0,p=.8 it is −log(.2)≈1.609. Mean over [N,1] yields scalar loss [].",
    shapes: [
      { name: "Logits", shape: "[N,1]", meaning: "Unbounded raw scores." },
      {
        name: "Probability",
        shape: "[N,1]",
        meaning: "Sigmoid(logits), clipped only for log safety.",
      },
      { name: "Target", shape: "[N,1]", meaning: "Float 0/1 labels." },
      { name: "Mean BCE", shape: "[]", meaning: "Scalar objective for backward." },
    ],
    diagram: {
      kind: "pipeline",
      title: "Probability and target become one scalar loss",
      description: "Select the loss stage for tensor shapes and the BCE equation.",
    },
    sections: sections(
      "Compute the manual formula carefully",
      [
        "For each target y∈{0,1} and probability p, BCE combines the positive term y log(p) with the negative term (1−y)log(1−p). Averaging over examples produces one scalar loss.",
        "log(0) is undefined. Clamp probabilities to [ε,1−ε] before the handwritten logarithms. Clipping is a guard for this teaching implementation, not a replacement for stable numerical methods.",
      ],
      "Prefer a fused logits loss in production",
      [
        "The source applies sigmoid and then the manual BCE. PyTorch's BCEWithLogitsLoss combines sigmoid and BCE using a numerically stable formulation, so it accepts logits directly and should be preferred for typical training.",
        "Do not pass sigmoid probabilities to BCEWithLogitsLoss. Conversely, BCELoss expects probabilities. Keep the target float and match the prediction shape exactly.",
      ],
    ),
    snippet: s[3],
    template: s[3],
    pitfalls: [
      "log(0) yields negative infinity; clip probabilities in the manual formula.",
      "BCEWithLogitsLoss expects logits, not sigmoid probabilities.",
      "BCELoss expects probabilities, not raw logits.",
      "Shape [B] may broadcast against [B,1] and silently form a [B,B] matrix.",
    ],
    production:
      "The manual implementation is pedagogical and less robust near extreme probabilities. Use BCEWithLogitsLoss for standard binary classification, with logits and float targets of matching shape.",
    revision: [
      "BCE is averaged to a scalar loss.",
      "Wrong confident predictions cost more.",
      "Clamp probabilities in a handwritten log formula.",
      "Use BCEWithLogitsLoss directly on logits in production.",
    ],
    questions: [
      qa("What is BCE when y=1 and p=.8?", "−log(.8), approximately .223."),
      qa("What shape should backward receive?", "A scalar loss, shape []."),
      qa("Which loss accepts raw logits?", "BCEWithLogitsLoss."),
      qa(
        "Why can [B] targets be dangerous beside [B,1] outputs?",
        "Broadcasting can create an unintended [B,B] comparison instead of matching pairs.",
      ),
    ],
  },
  {
    id: "manual-training-loop",
    moduleId: "training",
    title: "Backward, update and clear: the manual training loop.",
    subtitle:
      "Follow gradients from the scalar loss into W and b, then make one controlled SGD step.",
    minutes: 14,
    source: {
      start: 8,
      end: 9,
      kind: "source-derived",
      note: "The complete tested classifier, synthetic data and two-iteration gradient check are reconstructions; validation handling is supplemental.",
    },
    objectives: [
      "Trace loss.backward() into the manual parameters.",
      "Update leaves under no_grad and clear gradients.",
      "Use the interactive pipeline to inspect every stage.",
    ],
    intuition:
      "The gradient measures how a small weight change affects the loss. Manual SGD steps opposite that slope; clearing old gradients makes the next iteration independent.",
    equation: "W ← W − η∂L/∂W      b ← b − η∂L/∂b",
    numerical:
      "For one step with η=.1, a weight with gradient .4 moves down by .04. Calling backward on a later loss adds its gradient to the stored value, so zeroing is necessary before a fresh step.",
    shapes: [
      {
        name: "Training batch",
        shape: "[N,30]",
        meaning: "Standardized float32 features.",
      },
      { name: "Loss", shape: "[]", meaning: "Mean BCE over the training rows." },
      {
        name: "Weight gradient",
        shape: "[30,1]",
        meaning: "Derivative for each feature coefficient.",
      },
      {
        name: "Updated W,b",
        shape: "[30,1], [1]",
        meaning: "Same parameter shapes, new values.",
      },
    ],
    diagram: {
      kind: "pipeline",
      title: "Explore the training pipeline",
      description:
        "Select each numbered stage to reveal shapes, equations, code and expected values; the learning-rate control shows update direction.",
    },
    sections: sections(
      "One epoch has four explicit jobs",
      [
        "First call model(X_train) to create logits and probabilities. Next compute one scalar BCE loss. Then loss.backward() applies the chain rule and stores gradients in the leaf tensors.",
        "Update W and b under torch.no_grad() so optimizer arithmetic is not recorded as a new graph. Clear .grad before the next iteration. The source uses in-place gradient zeroing; setting gradients to None is another common pattern.",
      ],
      "Inspect both iterations and the full workflow",
      [
        "The gradient demo intentionally runs backward twice: 4 becomes 7 because 3 is added. Clearing then produces 3 on the next pass. This accumulation can be useful across microbatches, but otherwise changes the intended update.",
        "The complete script trains and tests on generated linearly separable data; it asserts decreasing training loss and correct shapes/dtypes. Its held-out test rows are not used to fit the scaler or choose the displayed threshold.",
      ],
    ),
    snippet: s[6],
    template: s[4],
    additionalSnippets: [s[5]],
    pitfalls: [
      "Updating tracked leaves outside no_grad can raise an in-place autograd error.",
      "Backward accumulates; clear gradients before independent iterations.",
      "Do not update parameters with stale gradients from a previous pass.",
      "Do not tune repeatedly against the held-out test split.",
    ],
    production:
      "For mini-batch training, call zero_grad before each independent batch. If intentionally accumulating microbatches, clear once at the start of the accumulation group and divide loss appropriately.",
    revision: [
      "Forward → scalar loss → backward → manual update.",
      "Autograd fills leaf .grad; it does not update parameter values.",
      "Use no_grad for the in-place SGD update.",
      "Clear gradients before the next independent step.",
    ],
    questions: [
      qa(
        "What does loss.backward compute?",
        "Gradients of the scalar loss with respect to tracked leaves such as weights and bias.",
      ),
      qa(
        "Why wrap parameter subtraction in no_grad?",
        "So the manual update is not recorded as part of the model's differentiable computation.",
      ),
      qa(
        "What is the gradient after derivatives 4 and 3 without clearing?",
        "7, because backward accumulates into .grad.",
      ),
      qa(
        "When might accumulation be intentional?",
        "When combining gradients from multiple microbatches before one parameter update.",
      ),
    ],
  },
  {
    id: "binary-evaluation",
    moduleId: "training",
    title: "Turn probabilities into predictions and inspect errors.",
    subtitle:
      "Apply the 0.5 threshold without dropping the batch axis; read TP, TN, FP and FN.",
    minutes: 9,
    source: {
      start: 9,
      end: 9,
      kind: "source-derived",
      note: "Confusion-count labels and explicit shape checks are supplemental interpretation.",
    },
    objectives: [
      "Apply threshold 0.5 to sigmoid probabilities.",
      "Compute accuracy with aligned [N,1] tensors.",
      "Interpret each confusion-matrix cell.",
    ],
    intuition:
      "A probability is a graded estimate; a threshold turns it into a decision. A confusion table then separates correct calls from two kinds of error.",
    equation: "ŷᵢ = 1[pᵢ ≥ .5]      accuracy = correct/N",
    numerical:
      "For probabilities [.9,.7,.4,.1] and labels [1,0,1,0], thresholding gives [1,1,0,0]. TP=1,TN=1,FP=1,FN=1; accuracy=.50.",
    shapes: [
      { name: "Probabilities", shape: "[N,1]", meaning: "Sigmoid outputs." },
      { name: "Predicted classes", shape: "[N,1]", meaning: "0/1 after threshold .5." },
      { name: "Targets", shape: "[N,1]", meaning: "Encoded truth labels." },
      { name: "Confusion counts", shape: "[] × 4", meaning: "TP, TN, FP and FN totals." },
    ],
    diagram: {
      kind: "pipeline",
      title: "Threshold, predict, and count outcomes",
      description:
        "Choose prediction stage to inspect the probability-to-class rule and output shapes.",
    },
    sections: sections(
      "Keep prediction and target aligned",
      [
        "The source thresholds sigmoid values at .5 and compares predicted labels with test labels. Reshape both sides consistently as [N,1]; avoid an unrestricted squeeze that removes the batch axis when N=1.",
        "Accuracy is the fraction of exact matches. It can hide class imbalance, so confusion counts show true positives, true negatives, false positives and false negatives separately.",
      ],
      "Interpret errors using the label direction",
      [
        "With B=0 and M=1, a false positive predicts malignant for a benign case; a false negative misses a malignant case. The cost of these errors depends on the application and is not captured by accuracy alone.",
        "The threshold is a decision rule, not part of the learned probability. Changing it should be guided by validation data and a stated error tradeoff; keep the test set for the final report.",
      ],
    ),
    snippet: s[7],
    template: s[7],
    pitfalls: [
      "Compare tensors with identical shapes to avoid unintended broadcasting.",
      "Do not squeeze away the batch axis for a one-row inference call.",
      "Accuracy can look high while the minority-class recall is poor.",
      "Read false positives and negatives relative to the documented B/M mapping.",
    ],
    production:
      "For clinical applications, this toy model is not suitable for diagnosis. Use validated data, calibrated evaluation, expert review and domain-specific error analysis.",
    revision: [
      "Threshold probability at .5 for the source demonstration.",
      "Keep prediction and target shape [N,1].",
      "Report TP/TN/FP/FN alongside accuracy.",
      "A threshold is a decision rule; test data remains untouched until final evaluation.",
    ],
    questions: [
      qa(
        "What does a false negative mean under M=1?",
        "A malignant case was predicted as benign.",
      ),
      qa(
        "Why use [N,1] rather than squeeze()?",
        "It preserves a stable output shape even when N=1.",
      ),
      qa(
        "What are the four confusion outcomes?",
        "True positive, true negative, false positive and false negative.",
      ),
      qa(
        "Should the test set choose the threshold?",
        "No. Use validation data for threshold choices and reserve the test set for final evaluation.",
      ),
    ],
  },
];
export const phase03Lessons = lessons;
export const phase03Snippets = s;
