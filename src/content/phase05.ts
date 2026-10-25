import type { Lesson, Snippet } from "./types";

const makeSnippet = (
  id: string,
  title: string,
  purpose: string,
  shapes: string,
  code: string,
  expected: string,
  dependencies = "Python 3.10+ · PyTorch · CPU · no downloads",
): Snippet => ({
  id,
  title,
  purpose,
  dependencies,
  shapes,
  code,
  expected,
  lines: code.split("\n").map((line, i) => {
    const text = line.trim();
    const note = text.startsWith("import ")
      ? "Import the dependency used in this example."
      : text.startsWith("class ")
        ? "Define the model or a small reusable helper."
        : text.startsWith("def ")
          ? "Keep one operation explicit and easy to check."
          : text.startsWith("assert ")
            ? "Verify a tensor contract or numerical invariant."
            : text.startsWith("print(")
              ? "Display the expected result."
              : text.startsWith("#")
                ? text.replace(/^#+\s*/, "")
                : `Trace the operation through ${shapes}.`;
    return `Line ${i + 1} · ${note}`;
  }),
});

const snippets: Snippet[] = [
  makeSnippet(
    "fashion-shapes",
    "Scale and flatten Fashion-MNIST images",
    "Check the image, pixel, and class-label contracts without downloading a dataset.",
    "images [B,1,28,28] → pixels [B,784], labels [B] long",
    `import torch

raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
pixels = raw.float() / 255.0
images = torch.zeros(4, 1, 28, 28)
flat = images.flatten(start_dim=1)
labels = torch.tensor([0, 1, 8, 9], dtype=torch.long)
assert pixels.min() >= 0 and pixels.max() <= 1
assert flat.shape == (4, 784) and labels.shape == (4,)
assert labels.dtype == torch.long
print(f"pixels={pixels.tolist()}, flat={list(flat.shape)}, labels={labels.dtype}")`,
    "pixels=[0.0, 0.5019608, 1.0], flat=[4, 784], labels=torch.int64",
  ),
  makeSnippet(
    "ann-architecture",
    "Inspect an ANN and its parameter count",
    "Build a 784 → 128 → 64 → 10 network and account for each Linear bias.",
    "[B,1,28,28] → [B,784] → [B,128] → [B,64] → [B,10]",
    `import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
)
logits = model(torch.zeros(5, 1, 28, 28))
parameters = sum(p.numel() for p in model.parameters())
expected = (784 + 1) * 128 + (128 + 1) * 64 + (64 + 1) * 10
assert logits.shape == (5, 10)
assert parameters == expected == 109386
print(f"logits={list(logits.shape)}, parameters={parameters}")`,
    "logits=[5, 10], parameters=109386",
  ),
  makeSnippet(
    "synthetic-ann-smoke",
    "Train and evaluate a tiny synthetic ANN",
    "Exercise multiclass batches, CrossEntropyLoss, optimizer updates, validation and a confusion matrix on CPU.",
    "images [N,1,28,28] → logits [B,10], labels [B] long",
    `import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(17)
device = torch.device("cpu")
labels = torch.arange(10).repeat_interleave(20)
images = torch.randn(200, 1, 28, 28) * 0.15
for label in range(10):
    images[labels == label, 0, 0, label] += 2.5
train_x, train_y = images[:150], labels[:150]
valid_x, valid_y = images[150:], labels[150:]
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=30, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=25)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
first_loss = loss_fn(model(train_x.to(device)), train_y.to(device)).item()
for _ in range(8):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
correct = total = 0
confusion = torch.zeros(10, 10, dtype=torch.long)
with torch.inference_mode():
    for batch_x, batch_y in valid_loader:
        logits = model(batch_x.to(device))
        predictions = logits.argmax(dim=1).cpu()
        correct += (predictions == batch_y).sum().item()
        total += batch_y.numel()
        confusion.index_put_((batch_y, predictions), torch.ones_like(batch_y), accumulate=True)
final_loss = loss_fn(model(train_x.to(device)), train_y.to(device)).item()
assert final_loss < first_loss and total == 50
assert confusion.sum().item() == total
print(f"train loss {first_loss:.3f} → {final_loss:.3f}; validation accuracy={correct / total:.2%}")`,
    "Training loss decreases; validation confusion counts sum to 50 examples.",
  ),
  makeSnippet(
    "cross-entropy-contract",
    "Check the multiclass loss contract",
    "Pass raw logits and integer class indices to CrossEntropyLoss; do not add softmax.",
    "logits [B,10] float, targets [B] long → scalar loss []",
    `import torch
from torch import nn

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.0]])
targets = torch.tensor([0, 1], dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits, targets)
probabilities = logits.softmax(dim=1)
assert logits.shape == (2, 3) and targets.shape == (2,)
assert targets.dtype == torch.long and loss.ndim == 0
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
print(f"loss={loss.item():.4f}, probabilities sum to one per row")`,
    "A finite scalar loss; each displayed probability row sums to 1.",
  ),
  makeSnippet(
    "metrics-confusion",
    "Compute accuracy, balanced accuracy and macro F1",
    "Derive class-level and overall metrics from a confusion matrix so imbalance is visible.",
    "confusion [C,C] → class recall/precision → aggregate scalars",
    `import torch

confusion = torch.tensor([[8, 2, 0], [1, 3, 1], [0, 2, 3]], dtype=torch.float64)
true_positive = confusion.diag()
support = confusion.sum(dim=1)
predicted = confusion.sum(dim=0)
recall = true_positive / support.clamp_min(1)
precision = true_positive / predicted.clamp_min(1)
f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
accuracy = true_positive.sum() / confusion.sum()
balanced_accuracy = recall.mean()
macro_f1 = f1.mean()
assert confusion.sum().item() == 20
assert 0 <= accuracy <= 1 and 0 <= balanced_accuracy <= 1 and 0 <= macro_f1 <= 1
print(f"accuracy={accuracy:.3f}, balanced_accuracy={balanced_accuracy:.3f}, macro_f1={macro_f1:.3f}")`,
    "accuracy=0.700, balanced_accuracy=0.711, macro_f1=0.695",
  ),
  makeSnippet(
    "epoch-metrics",
    "Aggregate loss by examples, not by batches",
    "Avoid giving a short final batch the same weight as a full batch when reporting epoch loss.",
    "batch mean losses × batch sizes → example-weighted epoch mean",
    `import torch

batch_losses = [torch.tensor(0.4), torch.tensor(1.0)]
batch_sizes = [4, 1]
weighted = sum(loss.item() * size for loss, size in zip(batch_losses, batch_sizes)) / sum(batch_sizes)
unweighted = sum(loss.item() for loss in batch_losses) / len(batch_losses)
assert abs(weighted - 0.52) < 1e-6
assert abs(unweighted - 0.70) < 1e-6
print(f"example-weighted={weighted:.2f}, misleading batch mean={unweighted:.2f}")`,
    "example-weighted=0.52, misleading batch mean=0.70",
  ),
  makeSnippet(
    "seed-training-curves",
    "Record train and validation loss by epoch",
    "Store split-specific mean losses to compare learning progress and spot a widening generalization gap.",
    "epoch → train loss, validation loss, validation accuracy",
    `import torch

history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
for train_loss, validation_loss, validation_accuracy in [
    (1.8, 1.7, 0.42), (1.1, 1.2, 0.66), (0.6, 1.4, 0.61)
]:
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
assert len(history["train_loss"]) == 3
assert history["train_loss"][-1] < history["train_loss"][0]
assert history["validation_loss"][-1] > history["validation_loss"][-2]
print("history recorded; final epoch shows a train/validation gap")`,
    "history recorded; final epoch shows a train/validation gap",
  ),
  makeSnippet(
    "device-placement",
    "Select a device and move every batch",
    "Keep model parameters, image batches and labels on the same device, with conditional transfer hints.",
    "CPU / CUDA / MPS → model, images [B,1,28,28], labels [B]",
    `import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model = nn.Linear(784, 10).to(device)
images = torch.zeros(4, 1, 28, 28).pin_memory() if device.type == "cuda" else torch.zeros(4, 1, 28, 28)
labels = torch.zeros(4, dtype=torch.long)
images = images.to(device, non_blocking=device.type == "cuda").flatten(start_dim=1)
labels = labels.to(device, non_blocking=device.type == "cuda")
logits = model(images)
assert logits.device == labels.device == next(model.parameters()).device
assert logits.shape == (4, 10)
print(f"device={device}, logits={list(logits.shape)}")`,
    "device=cpu (or available cuda/mps device), logits=[4, 10]",
  ),
  makeSnippet(
    "checkpoint-seed",
    "Seed a run and round-trip model weights",
    "Make initialization repeatable and save only the model state dictionary.",
    "seed → model parameters → state_dict CPU checkpoint",
    `import io
import torch
from torch import nn

torch.manual_seed(29)
first = nn.Linear(4, 2)
torch.manual_seed(29)
second = nn.Linear(4, 2)
assert torch.equal(first.weight, second.weight)
buffer = io.BytesIO()
torch.save(first.state_dict(), buffer)
buffer.seek(0)
restored = nn.Linear(4, 2)
restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
assert torch.equal(first.weight, restored.weight)
print("seeded initialization and weights-only checkpoint round-trip passed")`,
    "seeded initialization and weights-only checkpoint round-trip passed",
  ),
  makeSnippet(
    "fashion-mnist-optional",
    "Train on Fashion-MNIST when you opt in",
    "Use torchvision's dataset helper with download disabled by default; explicitly enable fetching if needed.",
    "Fashion-MNIST [N,1,28,28] → flattened batches → 10 logits",
    `"""Optional real-data exercise. Set DOWNLOAD_DATA=True to fetch missing files."""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DOWNLOAD_DATA = False  # Change to True only when you want torchvision to download the dataset.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()  # Converts uint8 pixels to float in [0, 1].
train_data = datasets.FashionMNIST("data", train=True, download=DOWNLOAD_DATA, transform=transform)
test_data = datasets.FashionMNIST("data", train=False, download=DOWNLOAD_DATA, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=device.type == "cuda")
test_loader = DataLoader(test_data, batch_size=128, pin_memory=device.type == "cuda")
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for images, labels in train_loader:
    images = images.to(device, non_blocking=device.type == "cuda")
    labels = labels.to(device, non_blocking=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(images), labels)  # Raw logits; no softmax before cross entropy.
    loss.backward()
    optimizer.step()
model.eval()
correct = total = 0
assert len(train_loader) > 0 and len(test_loader) > 0
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
print(f"test accuracy={correct / total:.2%} over {total} images")`,
    "Prints accuracy for one illustrative training pass and the test set; results vary.",
    "Python 3.10+ · PyTorch · torchvision · optional Fashion-MNIST download",
  ),
];

const byId = Object.fromEntries(snippets.map((snippet) => [snippet.id, snippet]));
const makeLesson = (args: {
  id: string;
  title: string;
  subtitle: string;
  source: [number, number];
  diagram: Lesson["diagram"];
  objective: string;
  intuition: string;
  equation: string;
  numerical: string;
  shapes: Lesson["shapes"];
  sections: Lesson["sections"];
  snippet: string;
  template: string;
  additional?: string[];
  pitfalls: string[];
  production: string;
  revision: string[];
  questions: [string, string][];
}): Lesson => {
  const snippet = byId[args.snippet];
  const template = byId[args.template];
  return {
    id: args.id,
    moduleId: "vision",
    title: args.title,
    subtitle: args.subtitle,
    minutes: 8,
    source: { start: args.source[0], end: args.source[1], kind: "source-derived" },
    objectives: [
      args.objective,
      "Trace the tensor shapes at each step.",
      "Run the accompanying CPU-safe checks before using a real dataset.",
    ],
    intuition: args.intuition,
    equation: args.equation,
    numerical: args.numerical,
    shapes: args.shapes,
    diagram: args.diagram,
    sections: args.sections,
    snippet,
    template,
    additionalSnippets: args.additional?.map((id) => byId[id]),
    pitfalls: args.pitfalls,
    production: args.production,
    revision: args.revision,
    questions: args.questions.map(([question, answer]) => ({ question, answer })),
  };
};

const sourceSection = (id: string, title: string, body: string) => ({
  id,
  title,
  paragraphs: body.split("\n"),
});

export const phase05Lessons: Lesson[] = [
  makeLesson({
    id: "fashion-mnist",
    title: "Fashion-MNIST gives an ANN ten labels to learn",
    subtitle: "From grayscale pixels to paired examples",
    source: [41, 42],
    diagram: {
      kind: "ann-data",
      title: "Split first, then batch paired images and labels",
      description:
        "Keep training, validation and test records disjoint. Flatten each [1,28,28] image per batch into 784 features while its integer class label stays paired.",
    },
    objective:
      "Represent a 28×28 grayscale image as 784 scaled features and keep its class as an integer target.",
    intuition:
      "Each image is a small grid of 784 brightness values. A dense network reads the flattened grid, while the label names one of ten clothing categories.",
    equation: "uint8 pixels / 255 → float [0,1] → flatten(1) [B,784]",
    numerical:
      "For B=32, a loader yields images [32,1,28,28] and labels [32]. Flattening from dimension 1 preserves the batch axis: [32,784].",
    shapes: [
      {
        name: "Image batch",
        shape: "[B,1,28,28]",
        meaning: "Grayscale channel and 28×28 pixels.",
      },
      {
        name: "Dense input",
        shape: "[B,784]",
        meaning: "One flattened feature vector per image.",
      },
      {
        name: "Labels",
        shape: "[B] long",
        meaning: "One class index from 0 through 9 per image.",
      },
    ],
    sections: [
      sourceSection(
        "representation",
        "Keep pixels and labels in the expected form",
        "Fashion-MNIST contains grayscale 28×28 images and ten clothing classes. ToTensor maps byte-valued pixels into floating point in [0,1]. A dense ANN needs the image flattened within each sample: images.flatten(start_dim=1) changes [B,1,28,28] to [B,784].\nCrossEntropyLoss class targets are integer class indices, stored as torch.long with shape [B]. Keep the image-label pair together when shuffling.",
      ),
      sourceSection(
        "splits",
        "Separate training, validation and test",
        "Train batches update the weights. Validation supports choices such as architecture and stopping time. Use the test split for a final report after choices are complete. The source notes a smaller training subset for speed; its score is not a guarantee for another split, seed, or training setup.",
      ),
    ],
    snippet: "fashion-shapes",
    template: "fashion-shapes",
    additional: ["fashion-mnist-optional"],
    pitfalls: [
      "Do not divide values already converted to [0,1] by 255 a second time.",
      "Do not flatten dimension 0; that merges examples in a batch.",
      "Do not cast class indices to float for CrossEntropyLoss.",
      "Do not repeatedly tune against the held-out test split.",
    ],
    production:
      "Record the class-to-index mapping with the model. Apply the same grayscale conversion, pixel scaling and flattening at inference time.",
    revision: [
      "Fashion-MNIST images are grayscale 28×28.",
      "Flatten from dimension 1 to preserve the batch axis.",
      "Cross-entropy class targets are long indices shaped [B].",
      "Keep train, validation and test roles separate.",
    ],
    questions: [
      [
        "What shape enters the first Linear layer for a batch of 32?",
        "[32,784], after flattening each image while preserving the batch dimension.",
      ],
      [
        "Why does the label tensor use torch.long?",
        "CrossEntropyLoss selects a class by integer index and expects class-index targets with dtype long.",
      ],
      [
        "What does ToTensor do for an 8-bit pixel?",
        "It converts the pixel to floating point and scales the 0–255 range to [0,1].",
      ],
      [
        "When should the test split guide architecture choices?",
        "It should not; use validation for choices and reserve test for the final report.",
      ],
    ],
  }),
  makeLesson({
    id: "ann-architecture",
    title: "A dense ANN turns 784 pixels into ten logits",
    subtitle: "Linear layers, ReLU and a tunable parameter budget",
    source: [42, 43],
    diagram: {
      kind: "ann",
      title: "Change hidden widths and see the parameter count",
      description:
        "Flatten each image to 784 features, pass it through two ReLU hidden layers, then produce ten class logits. The sliders recalculate weights and biases.",
    },
    objective:
      "Trace the 784 → H1 → H2 → 10 shape path and calculate every Linear weight and bias.",
    intuition:
      "Each Linear layer combines its inputs; ReLU lets hidden layers compose non-linear features. The last ten values are competing class scores.",
    equation: "params = (784+1)H₁ + (H₁+1)H₂ + (H₂+1)10",
    numerical:
      "With H₁=128 and H₂=64, the network has 109,386 trainable values. Its output is [B,10]; CrossEntropyLoss consumes those raw scores directly.",
    shapes: [
      { name: "Flatten", shape: "[B,784]", meaning: "Pixel features." },
      {
        name: "Hidden layers",
        shape: "[B,H1] → [B,H2]",
        meaning: "Dense features with ReLU.",
      },
      { name: "Class scores", shape: "[B,10]", meaning: "One raw logit for each class." },
    ],
    sections: [
      sourceSection(
        "layers",
        "Count weights and biases",
        "A Linear(in_features, out_features) layer stores a matrix with in_features × out_features values plus one bias per output. The interactive diagram sums these values for both hidden layers and the classifier.\nHidden width is a capacity and memory choice. It is a design parameter, not evidence that a wider network will generalize better.",
      ),
      sourceSection(
        "logits",
        "Return raw logits from the model",
        "The final Linear layer returns unnormalized scores called logits. CrossEntropyLoss applies the numerically stable log-softmax and negative log likelihood internally. Do not put Softmax in the model when that loss is used. Apply softmax only when normalized class probabilities are needed for display or downstream logic.",
      ),
    ],
    snippet: "ann-architecture",
    template: "ann-architecture",
    pitfalls: [
      "Do not add Softmax before CrossEntropyLoss.",
      "Include a bias term for each Linear output when counting parameters.",
      "Flatten only non-batch dimensions.",
      "More hidden units increase memory and compute as well as parameter count.",
    ],
    production:
      "Use the same feature order and model widths in training and inference. Save the exact architecture configuration alongside state_dict weights.",
    revision: [
      "A 28×28 grayscale image contains 784 values.",
      "Each Linear layer contributes weight and bias parameters.",
      "ReLU adds non-linearity between hidden Linear layers.",
      "Return ten logits for ten classes; pass them raw to cross entropy.",
    ],
    questions: [
      [
        "How many parameters are in Linear(784,128)?",
        "(784+1)×128 = 100,480 weights and biases.",
      ],
      [
        "What does a [B,10] output represent?",
        "Ten unnormalized class scores per example, one logit for each class.",
      ],
      [
        "Why omit Softmax before CrossEntropyLoss?",
        "CrossEntropyLoss expects logits and performs the stable log-softmax internally.",
      ],
      [
        "What is the parameter count for widths 128 and 64?",
        "(784+1)×128 + (128+1)×64 + (64+1)×10 = 109,386.",
      ],
    ],
  }),
  makeLesson({
    id: "multiclass-training",
    title: "Train a multiclass classifier with logits and integer labels",
    subtitle: "Cross-entropy, batches and explicit train/eval modes",
    source: [43, 50],
    diagram: {
      kind: "ann-data",
      title: "Train updates weights; validation selects; test reports",
      description:
        "Only shuffled training batches step the optimizer. Validation and test use eval mode and inference_mode with no gradient updates.",
    },
    objective:
      "Implement the multiclass optimization loop and respect the logits, target and evaluation contracts.",
    intuition:
      "For each image, cross entropy compares the correct class score to all alternatives. Backpropagation adjusts the network to increase the correct score relative to competing scores.",
    equation: "zero_grad → logits=model(images) → CE(logits, labels) → backward → step",
    numerical:
      "A batch of 32 produces logits [32,10] and labels [32]. CrossEntropyLoss returns one scalar mean loss when using its default reduction.",
    shapes: [
      {
        name: "Images",
        shape: "[B,1,28,28] float",
        meaning: "Scaled grayscale image batch.",
      },
      { name: "Logits", shape: "[B,10] float", meaning: "Unnormalized scores." },
      { name: "Targets", shape: "[B] long", meaning: "Correct class indices." },
    ],
    sections: [
      sourceSection(
        "training",
        "Make one optimizer update per training batch",
        "Call model.train() for the training pass. For each paired batch, move images and labels to the selected device, clear old gradients, compute logits and scalar cross entropy, backpropagate, then step the optimizer.\nThe source uses a particular subset, learning rate and long epoch count. Treat those as one experiment; use validation trends and a practical budget to choose a stopping point instead of copying a fixed 100 epochs.",
      ),
      sourceSection(
        "evaluation",
        "Evaluation mode and gradient mode are separate",
        "Call model.eval() so layers such as dropout or batch normalization use evaluation behavior, then disable gradient recording with torch.inference_mode(). Count correct predictions and examples over the complete split. Switch back to model.train() before the next training epoch.",
      ),
    ],
    snippet: "synthetic-ann-smoke",
    template: "cross-entropy-contract",
    additional: ["fashion-mnist-optional"],
    pitfalls: [
      "Pass logits [B,C] and class indices [B], not one-hot targets, to this loss mode.",
      "Do not apply softmax before CrossEntropyLoss.",
      "Call both model.eval() and inference_mode() for validation.",
      "Count examples across batches; a short final batch must not get full-batch weight.",
    ],
    production:
      "Log the loss and sample counts for each split. Keep the chosen checkpoint and run configuration so evaluation can be reproduced.",
    revision: [
      "Training uses zero_grad, forward, loss, backward and step.",
      "CrossEntropyLoss consumes logits and long targets.",
      "Use train mode for optimization and eval mode for evaluation.",
      "Inference mode avoids building an autograd graph during evaluation.",
    ],
    questions: [
      [
        "What are the expected input and target shapes?",
        "Logits [B,10] and class indices [B] with dtype torch.long.",
      ],
      [
        "When is model.train() used?",
        "Before training updates, to enable training-time module behavior.",
      ],
      [
        "Does eval() disable gradient calculation?",
        "No. eval changes module behavior; inference_mode separately disables autograd tracking.",
      ],
      [
        "How should total epoch accuracy be computed?",
        "Sum correct predictions and divide by the total example count across all batches.",
      ],
    ],
  }),
  makeLesson({
    id: "ann-evaluation",
    title: "Read curves and confusion counts together",
    subtitle: "Accuracy is useful, but class errors carry more detail",
    source: [44, 50],
    diagram: {
      kind: "metrics",
      title: "A confusion matrix exposes class-specific misses",
      description:
        "Rows are true classes and columns are predicted classes. Diagonal cells are correct predictions; off-diagonal cells show which labels are confused.",
    },
    objective:
      "Interpret validation curves and derive per-class metrics from predictions without treating one score as the whole evaluation.",
    intuition:
      "Accuracy says how many predictions matched. A confusion matrix shows where the errors went, which matters when some classes are rare or costly to miss.",
    equation: "accuracy = Σ diagonal / N · balanced accuracy = mean(class recall)",
    numerical:
      "In a 3-class matrix, a correct prediction lies on the diagonal. Row-normalized values estimate recall per true class; column totals describe predicted-class volume.",
    shapes: [
      {
        name: "Predictions",
        shape: "[N] long",
        meaning: "Predicted class for each example.",
      },
      {
        name: "Confusion matrix",
        shape: "[C,C]",
        meaning: "True-class rows by predicted-class columns.",
      },
      {
        name: "Curves",
        shape: "[epoch]",
        meaning: "Train and validation loss/metric history.",
      },
    ],
    sections: [
      sourceSection(
        "curves",
        "Compare train and validation history",
        "Plot both training and validation loss over epochs. Training loss falling while validation loss rises can indicate overfitting; curves should be read with the split sizes and metric definition. Keep the test split out of repeated model selection.\nReported accuracy in the source belongs to its specific subset, seed, epoch count and optimization settings. It is an observation, not a promised result for a different run.",
      ),
      sourceSection(
        "confusion",
        "Inspect errors and class support",
        "For a multiclass confusion matrix, rows represent true labels and columns predicted labels. The diagonal contains correct predictions. Per-class recall divides a diagonal count by its row support; per-class precision divides by the corresponding column total.\nBalanced accuracy averages class recalls. Macro F1 first computes F1 per class then averages classes equally. These summaries can be more informative with imbalance, but they do not replace looking at support, the matrix and the task's error costs.",
      ),
    ],
    snippet: "metrics-confusion",
    template: "metrics-confusion",
    additional: ["epoch-metrics", "seed-training-curves"],
    pitfalls: [
      "Do not infer model quality from training accuracy alone.",
      "Do not average per-batch accuracy equally when batch sizes differ.",
      "Macro averages treat classes equally; report class support as context.",
      "Do not select the model repeatedly using test metrics.",
    ],
    production:
      "Store split definitions, class names, confusion counts, metric averaging rules and the checkpoint identifier with each evaluation report.",
    revision: [
      "Rows in the confusion matrix are true classes.",
      "Balanced accuracy averages per-class recall.",
      "Macro F1 averages class F1 values equally.",
      "Validation supports choices; test is reserved for final reporting.",
    ],
    questions: [
      [
        "What does an off-diagonal cell mean?",
        "An example of the row's true class was predicted as the column's class.",
      ],
      [
        "How does balanced accuracy differ from ordinary accuracy?",
        "It averages recall per class, giving each class equal weight rather than each example.",
      ],
      [
        "What can a train/validation loss gap suggest?",
        "The model may be overfitting, especially if training loss improves while validation loss worsens.",
      ],
      [
        "Why include class support with macro metrics?",
        "Equal-weighted summaries hide how many examples each class contributes and may be uncertain for small classes.",
      ],
    ],
  }),
  makeLesson({
    id: "device-placement",
    title: "Place the model and every batch on one device",
    subtitle: "CUDA, Apple MPS and CPU fallback",
    source: [50, 54],
    diagram: {
      kind: "device",
      title: "Choose an available accelerator, otherwise use CPU",
      description:
        "The model parameters and both batch tensors must share one device. Transfers happen at the input boundary; CPU remains a valid fallback.",
    },
    objective:
      "Select cuda, mps or cpu safely and move model inputs and targets consistently.",
    intuition:
      "A device is where a tensor's storage and operations live. A model on an accelerator cannot multiply a CPU batch until the batch is moved there too.",
    equation: "model.to(device); images.to(device); labels.to(device)",
    numerical:
      "For a CUDA DataLoader, pin_memory=True requests page-locked host batches; non_blocking=True can make host-to-device copies asynchronous when conditions permit. Neither option removes the copy.",
    shapes: [
      {
        name: "Model parameters",
        shape: "device D",
        meaning: "Moved once before the loop.",
      },
      {
        name: "Images",
        shape: "[B,1,28,28] on D",
        meaning: "Moved before the forward pass.",
      },
      {
        name: "Labels",
        shape: "[B] long on D",
        meaning: "Moved before loss calculation.",
      },
    ],
    sections: [
      sourceSection(
        "selection",
        "Select an available device",
        "Prefer CUDA when available, then Apple MPS where supported, and fall back to CPU. Check availability before choosing a backend, and move the model once before training. A correct CPU path keeps examples runnable on machines with no accelerator.",
      ),
      sourceSection(
        "transfer",
        "Move inputs and targets with the model",
        "Move each image batch and its labels to the same device as the parameters. Pinned memory is page-locked host memory that can improve transfer behavior for CUDA workloads; it is not accelerator memory and it does not skip host-to-device transfer.\nUse non_blocking transfers only as a transfer hint where the source and backend can support it. Measure end-to-end throughput before increasing workers, batch size, or memory use. Handle out-of-memory errors by reducing batch size or model size and releasing unused references.",
      ),
    ],
    snippet: "device-placement",
    template: "device-placement",
    additional: ["checkpoint-seed"],
    pitfalls: [
      "Do not leave labels on CPU while logits are on an accelerator.",
      "Pinned memory is host memory, not GPU memory.",
      "non_blocking=True does not eliminate host-to-device copies.",
      "Do not assume CUDA exists on every system; keep a tested CPU fallback.",
    ],
    production:
      "Log the selected device and peak memory. Keep inference device selection configurable and validate outputs on the deployment backend.",
    revision: [
      "Move model parameters before the training loop.",
      "Move image batches and labels to the same device.",
      "Pinning affects host memory and can help CUDA transfers.",
      "Keep a CPU fallback and measure before tuning transfer settings.",
    ],
    questions: [
      [
        "Why must labels move to the model's device?",
        "The loss operation compares logits and targets and requires compatible device placement.",
      ],
      [
        "What does pin_memory do?",
        "It asks the loader for page-locked host memory, which can aid CUDA transfer throughput.",
      ],
      [
        "Does non_blocking=True remove the data transfer?",
        "No. It requests asynchronous transfer behavior where supported; the copy still occurs.",
      ],
      [
        "What fallback should an example retain?",
        "A CPU path when CUDA or MPS is unavailable.",
      ],
    ],
  }),
];

export const phase05Snippets = snippets;
