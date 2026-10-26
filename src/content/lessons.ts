import type { Lesson, Module, Snippet } from "./types";
import { phase02Lessons } from "./phase02";
import { phase03Lessons } from "./phase03";
import { phase04Lessons } from "./phase04";
import { phase05Lessons } from "./phase05";
import { phase06Lessons } from "./phase06";
import { phase07Lessons } from "./phase07";
import { phase08Lessons } from "./phase08";
import { phase09Lessons } from "./phase09";
import { phase10Lessons } from "./phase10";
import { phase11Lessons } from "./phase11";
import { phase12Lessons } from "./phase12";
import coverage from "../../docs/source-coverage.json";

const derivative: Snippet = {
  id: "scalar-gradient",
  title: "Your first backward pass",
  purpose: "Differentiate a scalar square and check the result.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "x [] → y [] → x.grad []",
  code: `import torch

x = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
y = x ** 2
y.backward()
assert x.grad is not None
assert torch.allclose(x.grad, torch.tensor(6.0))
print(f"value={y.item():.1f}, gradient={x.grad.item():.1f}")`,
  expected: "value=9.0, gradient=6.0",
  lines: [
    "1 · Import PyTorch; no dataset or accelerator is needed.",
    "3 · Create a floating-point leaf tensor and ask autograd to track its use.",
    "4 · Square the value. This creates a new tensor with a backward operation.",
    "5 · Seed the scalar output with derivative 1 and apply the chain rule backward.",
    "6–7 · Check the leaf gradient exists and equals the analytical derivative.",
    "8 · Convert scalar tensors to Python numbers for readable output.",
  ],
};
const derivativeTemplate: Snippet = {
  id: "derivative-check",
  title: "Reusable derivative check",
  purpose: "Compare a differentiable function with its analytical derivative.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "input [] → scalar output []",
  code: `import torch

def check_derivative(fn, analytic, value):
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    fn(x).backward()
    expected = analytic(x.detach())
    assert torch.allclose(x.grad, expected, atol=1e-5)
    return x.grad.item()

print(check_derivative(lambda x: x ** 2, lambda x: 2 * x, 3.0))`,
  expected: "6.0",
  lines: [
    "1 · Import the only dependency.",
    "3 · Accept a scalar function, its derivative, and an evaluation point.",
    "4 · Use a fresh leaf so earlier gradients cannot accumulate.",
    "5 · Evaluate the function and differentiate its scalar result.",
    "6 · Detach the reference calculation from the graph.",
    "7–8 · Assert agreement within float32 tolerance, then return a Python scalar.",
    "10 · Supply f(x) = x² and its derivative 2x.",
  ],
};
const step: Snippet = {
  id: "manual-step",
  title: "One gradient descent step",
  purpose: "Move a scalar weight toward a target using its gradient.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "w [] → loss [] → w.grad []",
  code: `import torch

w = torch.tensor(3.0, dtype=torch.float32, requires_grad=True)
before = (w - 1.0) ** 2
before.backward()
assert torch.allclose(w.grad, torch.tensor(4.0))
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad.zero_()
after = (w - 1.0) ** 2
assert after.item() < before.item()
assert torch.allclose(w, torch.tensor(2.6))
print(f"weight={w.item():.2f}, loss={after.item():.2f}")`,
  expected: "weight=2.60, loss=2.56",
  lines: [
    "1 · Import PyTorch.",
    "3 · Make a trainable scalar starting at 3.",
    "4–6 · Evaluate squared error against 1 and check its derivative is 4.",
    "7–8 · Update the parameter without recording the optimizer arithmetic.",
    "9 · Clear the gradient accumulated on this leaf.",
    "10 · Recompute the loss at the updated parameter.",
    "11–12 · Verify lower loss and the expected parameter value.",
    "13 · Report the new value, not the stale pre-update loss.",
  ],
};
const stepTemplate: Snippet = {
  id: "manual-loop",
  title: "Reusable scalar optimization loop",
  purpose: "Repeat the forward, backward, update and clear lifecycle.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "parameter [] → loss []",
  code: `import torch

def fit_scalar(initial=3.0, target=1.0, lr=0.1, steps=20):
    w = torch.tensor(initial, dtype=torch.float32, requires_grad=True)
    for _ in range(steps):
        loss = (w - target) ** 2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad = None
    return w.detach()

result = fit_scalar()
assert abs(result.item() - 1.0) < 0.03
print(f"weight={result.item():.3f}")`,
  expected: "weight=1.023",
  lines: [
    "1 · Import PyTorch.",
    "3 · Expose initialization, target, step size and iteration count.",
    "4 · Create the floating-point leaf parameter.",
    "5–7 · Rebuild the scalar loss graph on every iteration and differentiate it.",
    "8–9 · Apply the manual update in a no-grad context.",
    "10 · Set the gradient to None so the next backward starts fresh.",
    "11 · Return a value without graph history.",
    "13–15 · Check the fitted parameter is close to the target and print it.",
  ],
};
const batches: Snippet = {
  id: "first-batches",
  title: "From ten samples to three batches",
  purpose: "Inspect the final partial batch without downloading data.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "features [10, 2] → batch [B, 2]; labels [B]",
  code: `import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
y = torch.arange(10, dtype=torch.long)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=4, shuffle=False)
sizes = [features.shape[0] for features, labels in loader]
assert sizes == [4, 4, 2]
assert len(loader) == 3
print(sizes)`,
  expected: "[4, 4, 2]",
  lines: [
    "1–2 · Import tensors, the paired-sample dataset, and the batch iterator.",
    "4 · Arrange twenty float values into ten two-feature rows.",
    "5 · Create one integer label per row. These are illustrative IDs.",
    "6 · Pair each feature row with its corresponding label.",
    "7 · Keep sequential ordering so the output is deterministic.",
    "8 · Iterate batches and collect their actual sample counts.",
    "9–11 · Assert that the last two samples survive and print the sizes.",
  ],
};
const batchTemplate: Snippet = {
  id: "batch-factory",
  title: "Reusable loader factory",
  purpose: "Control ordering and final-batch handling explicitly.",
  dependencies: "Python 3.10+ · PyTorch · CPU",
  shapes: "features [N, D] + targets [N] → [B, D] + [B]",
  code: `import torch
from torch.utils.data import TensorDataset, DataLoader

def make_loader(x, y, batch_size=4, training=True, drop_last=False):
    assert len(x) == len(y)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=training, drop_last=drop_last,
                      generator=generator, num_workers=0)

x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
y = torch.arange(10, dtype=torch.long)
loader = make_loader(x, y, drop_last=True)
assert [len(a) for a, b in loader] == [4, 4]
print("2 full batches; 2 samples dropped")`,
  expected: "2 full batches; 2 samples dropped",
  lines: [
    "1–2 · Import the tensor and batching tools.",
    "4–5 · Define the loader settings and verify sample alignment.",
    "6 · Seed the sampler for repeatable toy shuffling.",
    "7–9 · Build the loader; a single process is portable on Windows.",
    "11–12 · Construct deterministic toy features and labels.",
    "13 · Opt in to dropping the partial final batch.",
    "14–15 · Check the actual number of retained batches and describe the tradeoff.",
  ],
};

export const lessons: Lesson[] = [
  {
    id: "autograd-intuition",
    moduleId: "foundations",
    title: "Let the graph do the calculus.",
    subtitle: "A small square, a backward pass, and the idea behind learning.",
    minutes: 8,
    source: {
      start: 1,
      end: 3,
      kind: "source-derived",
      note: "Fresh examples, assertions and the reusable check are supplemental.",
    },
    objectives: [
      "Interpret a gradient as local sensitivity, not an update.",
      "Trace values forward and derivatives backward.",
      "Read a leaf tensor’s .grad after backward().",
    ],
    intuition:
      "Imagine a chain of gears. Turning the first gear changes the last one. To find how sensitive the final gear is to the first, multiply the local turning ratios along the chain. Autograd keeps the record of those local ratios for you.",
    equation: "y = x²        dy/dx = 2x",
    numerical:
      "At x = 3, y = 9 and dy/dx = 6. A small increase of 0.01 in x gives 3.01² = 9.0601: the change is approximately 6 × 0.01 = 0.06. The extra 0.0001 is the second-order term.",
    shapes: [
      {
        name: "Input x",
        shape: "[]",
        meaning: "One float32 scalar; a leaf with tracking enabled.",
      },
      {
        name: "Square y",
        shape: "[]",
        meaning: "One scalar result with a grad_fn.",
      },
      {
        name: "Gradient x.grad",
        shape: "[]",
        meaning: "One derivative, 6.0, stored after backward.",
      },
    ],
    diagram: {
      kind: "graph",
      title: "Values forward. Sensitivity backward.",
      description:
        "A scalar leaf x passes through a square operation to y. Forward values follow the solid arrow; the derivative returns along the dashed arrow.",
    },
    sections: [
      {
        id: "understand",
        title: "What a gradient actually tells you",
        paragraphs: [
          "Neural networks are compositions of tensor operations. Training asks how a loss would change if a parameter moved slightly. The gradient answers that local question. An optimizer then decides how far and in which direction to move.",
          "Here the entire model is just a square. Its derivative is positive at x = 3, so increasing x increases y locally. At negative x, the derivative is negative. A zero derivative describes local flatness; it does not by itself prove that you found a minimum.",
        ],
      },
      {
        id: "mechanics",
        title: "Build once, then walk backward",
        paragraphs: [
          "Setting requires_grad=True on a floating-point leaf asks PyTorch to record differentiable operations that depend on it. Evaluating x ** 2 builds a result connected to that leaf. Calling y.backward() starts with dy/dy = 1 and propagates derivatives toward the leaf.",
          "The derivative is stored in x.grad. Intermediate results do not retain .grad by default. Backward computes derivatives; it does not change x. Our diagram is a JavaScript calculation of this simple function, while the Python examples must be run in Python.",
        ],
      },
    ],
    snippet: derivative,
    template: derivativeTemplate,
    pitfalls: [
      "Integer tensors cannot require gradients. Use an explicit floating-point dtype.",
      "Reading .grad before backward returns None. It is not a numerical zero.",
      "A vector output needs a scalar reduction or an explicit gradient argument.",
      "A second backward adds to existing leaf gradients. Clear them between independent optimization steps.",
    ],
    production:
      "For a full model, keep the loss scalar, avoid in-place edits of trainable leaves in the recorded forward pass, and let an optimizer manage parameters. See the tensor and autograd lessons for the complete workflow.",
    revision: [
      "Forward computes values and records a graph.",
      "Backward applies the chain rule and accumulates gradients.",
      "requires_grad enables tracking; .grad is the accumulated derivative.",
      "A derivative is sensitivity. An optimizer makes the update.",
    ],
    questions: [
      {
        question: "What are y and x.grad when x = −2?",
        answer: "y = 4 and x.grad = −4. The derivative is 2x, including its sign.",
      },
      {
        question: "Does y.backward() update the value of x?",
        answer:
          "No. It computes and accumulates gradients. A separate manual update or optimizer step changes parameters.",
      },
      {
        question: "Why can the scalar output omit a gradient argument?",
        answer:
          "Backward implicitly seeds that scalar output with 1: the derivative of y with respect to itself.",
      },
      {
        question: "What does x.grad = None before backward mean?",
        answer:
          "No gradient is currently stored on x. It does not claim the mathematical derivative equals zero.",
      },
    ],
  },
  {
    id: "one-training-step",
    moduleId: "training",
    title: "One step toward a smaller loss.",
    subtitle: "Separate calculating a gradient from using it to learn.",
    minutes: 9,
    source: {
      start: 8,
      end: 9,
      kind: "source-derived",
      note: "Manual updates also appear on PDF pp. 13–14. The scalar squared-error example and template are supplemental; the source lab uses binary cross entropy.",
    },
    objectives: [
      "Follow forward → loss → backward → update.",
      "Calculate a gradient descent update by hand.",
      "Explain why updates use no_grad and why gradients must be cleared.",
    ],
    intuition:
      "Picture standing on a smooth hillside. The gradient points uphill at your current position. A small step in the opposite direction can take you downhill. The learning rate is your stride length: too large a stride can carry you across the valley.",
    equation: "L(w) = (w − 1)²      w_new = w − η · 2(w − 1)",
    numerical:
      "Start at w = 3. Loss is (3 − 1)² = 4 and the gradient is 4. With learning rate η = 0.1, w_new = 3 − 0.1 × 4 = 2.6. Recomputed loss is (2.6 − 1)² = 2.56.",
    shapes: [
      {
        name: "Parameter w",
        shape: "[]",
        meaning: "Scalar trainable weight initialized to 3.",
      },
      {
        name: "Loss L",
        shape: "[]",
        meaning: "Scalar squared distance from the target 1.",
      },
      {
        name: "Gradient ∂L/∂w",
        shape: "[]",
        meaning: "Scalar 4 before the first update.",
      },
      {
        name: "Updated weight",
        shape: "[]",
        meaning: "Same leaf parameter; new value 2.6.",
      },
    ],
    diagram: {
      kind: "update",
      title: "Choose a stride. Inspect the step.",
      description:
        "Squared-error curve centered on target 1. The old parameter is 3. A learning-rate slider moves the new parameter according to w minus learning rate times gradient.",
    },
    sections: [
      {
        id: "understand",
        title: "The four jobs in a training step",
        paragraphs: [
          "The source training pipeline uses a Wisconsin breast cancer binary classifier. It computes predictions, measures binary cross entropy, calls backward, and manually subtracts learning-rate-scaled gradients from its weights and bias. This preview isolates the last two ideas with a scalar squared-error loss.",
          "First compute the loss at the current parameter. Next compute its derivative. Only then update the parameter. Finally clear the stored derivative so the next iteration uses fresh information. Each iteration builds a new computation graph.",
        ],
      },
      {
        id: "mechanics",
        title: "Make the update outside the graph",
        paragraphs: [
          "The parameter update is an optimization action, not part of the prediction function being differentiated. Wrap the in-place subtraction in torch.no_grad() to avoid recording that arithmetic and to permit the leaf update. The parameter still requires gradients in the next forward pass.",
          "For this particular quadratic, the error after a step is multiplied by (1 − 2η). A constant 0 < η < 1 shrinks its magnitude. At η = 1 it oscillates without improving; larger values diverge. Real neural-network loss surfaces do not share this simple global guarantee.",
        ],
      },
    ],
    snippet: step,
    template: stepTemplate,
    pitfalls: [
      "Do not update a leaf in-place in normal gradient-tracking mode.",
      "Do not mistake the old loss tensor for a recomputed post-update loss.",
      "Without clearing .grad, independent iterations accumulate derivatives.",
      "A very large learning rate can increase loss; the negative-gradient direction is a local guide.",
    ],
    production:
      "In practical nn.Module code, use optimizer.zero_grad(set_to_none=True), loss.backward(), and optimizer.step(). The manual classification lessons show preprocessing, loss choices, and a complete training workflow.",
    revision: [
      "Backward calculates; the update changes parameters.",
      "Subtract the gradient to move locally downhill.",
      "Use no_grad for a manual in-place update.",
      "Clear gradients and rebuild the loss each iteration.",
    ],
    questions: [
      {
        question: "What is w_new for w = 3 and η = 0.25?",
        answer: "The gradient is 4, so w_new = 3 − 0.25 × 4 = 2. The new loss is 1.",
      },
      {
        question: "What happens at η = 1 for this example?",
        answer:
          "The new weight is −1 and the loss is still 4. Repeated steps oscillate across the target.",
      },
      {
        question: "Why is the update wrapped in no_grad?",
        answer:
          "It avoids tracking optimizer arithmetic and allows the trainable leaf to be changed in-place.",
      },
      {
        question: "Does no_grad permanently turn off requires_grad?",
        answer:
          "No. It changes recording within the context. The next forward pass can build a fresh graph.",
      },
    ],
  },
  {
    id: "mini-batches",
    moduleId: "data-pipeline",
    title: "Small batches. A steady pipeline.",
    subtitle: "How individual samples become the tensors your model sees.",
    minutes: 8,
    source: {
      start: 25,
      end: 28,
      kind: "source-derived",
      note: "Implicit method calls are elaborated on PDF pp. 39–40. TensorDataset toy data and deterministic templates are supplemental.",
    },
    objectives: [
      "Distinguish a Dataset from a DataLoader.",
      "Trace indices through sample lookup and collation.",
      "Count full and partial batches, with and without drop_last.",
    ],
    intuition:
      "A library catalogue tells you where each book lives. A librarian chooses a list of catalogue numbers, fetches the books, and puts them on a trolley. The Dataset is the lookup interface; the sampler selects indices; the DataLoader coordinates fetching and collation into a batch.",
    equation: "batches = ceil(N / B)      drop_last: floor(N / B)",
    numerical:
      "For N = 10 samples and batch size B = 4, the batch sizes are 4, 4, 2: three updates per full pass if you train once per batch. With drop_last=True, there are two full batches and two omitted samples.",
    shapes: [
      {
        name: "Stored features",
        shape: "[10, 2]",
        meaning: "Ten examples, each containing two numeric features.",
      },
      {
        name: "One sample",
        shape: "[2] + []",
        meaning: "A feature row and its scalar label.",
      },
      {
        name: "Full batch",
        shape: "[4, 2] + [4]",
        meaning: "Four rows stacked together with four labels.",
      },
      {
        name: "Last batch",
        shape: "[2, 2] + [2]",
        meaning: "Only two samples remain when drop_last=False.",
      },
    ],
    diagram: {
      kind: "batch",
      title: "Ten samples, grouped your way.",
      description:
        "Ten numbered samples are partitioned into batches of the selected size. Dropped samples are explicitly marked when drop_last is enabled.",
    },
    sections: [
      {
        id: "understand",
        title: "One object knows the samples. Another serves batches.",
        paragraphs: [
          "For a map-style Dataset, __len__ reports the sample count and __getitem__(index) retrieves one sample. TensorDataset already supplies those methods for tensors with matching leading dimensions. The DataLoader uses a sampler to choose indices, fetches samples and calls a collate function to combine them.",
          "Default collation stacks equal-shaped tensors. Ten feature rows of shape [2] can become a batch tensor [4, 2], then another [4, 2], then [2, 2]. Variable-length text needs a custom collation strategy, such as padding, which is covered later.",
        ],
      },
      {
        id: "mechanics",
        title: "Be deliberate about ordering and the last batch",
        paragraphs: [
          "Training usually shuffles samples between passes. Evaluation commonly keeps a predictable order; accuracy itself is independent of ordering if predictions stay aligned with labels. This example keeps shuffle=False so you can inspect exact batches.",
          "Dropping the last batch can help when a training operation needs more than one example, but it discards data for that pass. Do not assume every batch has your requested size. These count formulas describe this map-style, single-process example; custom samplers and iterable datasets can have different length semantics.",
        ],
      },
    ],
    snippet: batches,
    template: batchTemplate,
    pitfalls: [
      "Features and labels must contain the same number of samples.",
      "Do not reshape every output to a fixed batch size: the last batch can be smaller.",
      "A larger batch does not universally mean faster convergence or better generalization.",
      "More workers are not always faster. Start at num_workers=0, particularly for portable Windows examples.",
    ],
    production:
      "Keep train, validation and test datasets separate. Fit learned preprocessing only on the training split. The data-loading lessons cover worker configuration, custom collation, sampling, and a complete offline classification pipeline.",
    revision: [
      "Dataset: retrieve one sample by index.",
      "Sampler: choose which indices to request.",
      "Collate: combine retrieved samples into batch tensors.",
      "drop_last=False retains a smaller final batch.",
    ],
    questions: [
      {
        question: "How many batches for N = 10, B = 3, drop_last=False?",
        answer: "Four batches, with sizes 3, 3, 3, 1.",
      },
      {
        question: "What changes if drop_last=True?",
        answer:
          "Only the three full batches are emitted. One sample is omitted for that pass.",
      },
      {
        question: "Which part stacks individual feature rows?",
        answer:
          "The collate function, invoked by the DataLoader. Default collation stacks equal-shaped tensors.",
      },
      {
        question: "Does shuffle=True separate labels from their features?",
        answer:
          "No. Indices fetch paired samples. Correctly implemented datasets keep each label with its feature row.",
      },
    ],
  },
  ...phase02Lessons,
  ...phase03Lessons,
  ...phase04Lessons,
  ...phase05Lessons,
  ...phase06Lessons,
  ...phase07Lessons,
  ...phase08Lessons,
  ...phase09Lessons,
  ...phase10Lessons,
  ...phase11Lessons,
  ...phase12Lessons,
];

export const modules: Module[] = [
  {
    id: "foundations",
    number: "01",
    title: "Foundations & autograd",
    description: "How tensors learn",
    lessons: [
      { id: "autograd-intuition", title: "Autograd intuition", ready: true },
      {
        id: "tensor-prerequisites",
        title: "Tensor essentials · supplemental",
        ready: true,
      },
      { id: "derivatives-chain-rule", title: "Derivatives & chain rule", ready: true },
      { id: "computation-graph", title: "Computation graph", ready: true },
      { id: "scalar-autograd", title: "Scalar autograd", ready: true },
      { id: "vector-gradients", title: "Vector gradients", ready: true },
      { id: "gradient-contexts", title: "Gradient control", ready: true },
    ],
  },
  {
    id: "training",
    number: "02",
    title: "The training loop",
    description: "From loss to learning",
    lessons: [
      { id: "one-training-step", title: "One training step", ready: true },
      {
        id: "binary-classification",
        title: "Classification & labels",
        ready: true,
      },
      { id: "preprocessing", title: "Split & preprocess", ready: true },
      { id: "manual-forward", title: "Manual neuron", ready: true },
      { id: "manual-loss", title: "Binary cross-entropy", ready: true },
      { id: "manual-training-loop", title: "Manual training pipeline", ready: true },
      { id: "binary-evaluation", title: "Threshold & evaluation", ready: true },
      { id: "manual-vs-idiomatic", title: "Manual vs idiomatic PyTorch", ready: true },
      { id: "nn-module", title: "Models with nn.Module", ready: true },
      {
        id: "sequential-architecture",
        title: "Sequential or explicit forward",
        ready: true,
      },
      { id: "loss-conventions", title: "Losses and target shapes", ready: true },
      { id: "optimizer-lifecycle", title: "The optimizer lifecycle", ready: true },
      {
        id: "idiomatic-classifier",
        title: "An idiomatic binary classifier",
        ready: true,
      },
    ],
  },
  {
    id: "data-pipeline",
    number: "03",
    title: "Data that flows",
    description: "Samples, batches, loaders",
    lessons: [
      { id: "mini-batches", title: "Mini-batch intuition", ready: true },
      { id: "custom-dataset", title: "Build a Dataset", ready: true },
      { id: "dataloader-pipeline", title: "Sampler to batch", ready: true },
      { id: "collation-workers", title: "Collation and worker options", ready: true },
      { id: "batched-classifier", title: "Train with DataLoader", ready: true },
    ],
  },
  ...[
    {
      id: "vision",
      title: "Learning to see",
      description: "ANN · GPU · regularization",
      ids: [
        "fashion-mnist",
        "ann-architecture",
        "multiclass-training",
        "ann-evaluation",
        "device-placement",
        "overfitting",
        "dropout",
        "batch-normalization",
        "weight-decay",
        "regularization-comparison",
      ],
    },
    {
      id: "tuning",
      title: "Better experiments",
      description: "Optuna & validation",
      ids: [
        "hyperparameter-search",
        "optuna-basics",
        "optuna-tabular",
        "optuna-samplers",
        "optuna-plots",
        "dynamic-search",
        "optuna-tracking",
        "ann-search-space",
        "ann-objective",
        "dynamic-ann",
      ],
    },
    {
      id: "convolution",
      title: "Convolutional networks",
      description: "Filters, features & shapes",
      ids: [
        "cnn-motivation", "image-tensors", "convolution-math", "convolution-shapes",
        "convolution-channels", "cnn-activations", "pooling", "receptive-field",
        "cnn-classifier",
      ],
    },
    {
      id: "transfer-learning",
      title: "Transfer learning",
      description: "VGG16 & preprocessing",
      ids: [
        "transfer-learning", "frozen-features", "vgg-preprocessing", "vgg-transform",
        "vgg-normalization", "vgg16-lab", "weights-transforms",
      ],
    },
    {
      id: "sequences",
      title: "Thinking in sequences",
      description: "RNNs & backpropagation",
      ids: ["rnn-motivation", "rnn-use-cases", "rnn-shapes", "recurrent-cell", "rnn-variants", "bptt", "gradient-pathologies", "faq-classifier", "faq-vocabulary", "faq-vocabulary-map", "faq-data", "faq-architecture", "faq-training", "faq-inference"],
    },
    {
      id: "memory",
      title: "Long short-term memory",
      description: "Gates & next-word prediction",
      ids: ["lstm-intuition", "lstm-states", "lstm-vectors", "lstm-gates", "lstm-blueprint", "lstm-numerical", "lstm-gradient-route", "lstm-pytorch-shapes", "lstm-parameters", "next-word-data", "next-word-tokenization", "next-word-vocabulary", "next-word-prefixes", "next-word-padding", "next-word-architecture", "next-word-training", "next-word-generation", "next-word-sampling", "next-word-project"],
    },
  ].map((m, i) => ({
    id: m.id,
    number: String(i + 4).padStart(2, "0"),
    title: m.title,
    description: m.description,
    lessons: m.ids.map((id) => ({
      id,
      title:
        coverage.blocks.flatMap((b) => b.topics).find((t) => t.lesson === id)?.topic ??
        id,
      ready:
        phase05Lessons.some((lesson) => lesson.id === id) ||
        phase06Lessons.some((lesson) => lesson.id === id) ||
        phase07Lessons.some((lesson) => lesson.id === id) ||
        phase08Lessons.some((lesson) => lesson.id === id) ||
        phase09Lessons.some((lesson) => lesson.id === id) ||
        phase10Lessons.some((lesson) => lesson.id === id) ||
        phase11Lessons.some((lesson) => lesson.id === id) ||
        phase12Lessons.some((lesson) => lesson.id === id) ||
        lessons.some((lesson) => lesson.id === id),
    })),
  })),
];
export const allSnippets = lessons.flatMap((l) => [
  l.snippet,
  l.template,
  ...(l.additionalSnippets ?? []),
]);
export const searchable = lessons.map((l) => ({
  id: l.id,
  title: l.title,
  path: modules.find((m) => m.id === l.moduleId)!.title,
  text: [
    l.title,
    l.subtitle,
    ...l.objectives,
    l.intuition,
    l.equation,
    l.numerical,
    ...l.sections.flatMap((s) => [s.title, ...s.paragraphs]),
    ...l.pitfalls,
    ...l.revision,
    l.snippet.code,
    l.template.code,
    ...(l.additionalSnippets ?? []).flatMap((snippet) => [
      snippet.title,
      snippet.purpose,
      snippet.dependencies,
      snippet.shapes,
      snippet.code,
      snippet.expected,
      ...snippet.lines,
    ]),
    ...l.questions.flatMap((q) => [q.question, q.answer]),
  ].join(" "),
}));
