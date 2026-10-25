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
  lines: code.split("\n").map((line, index) => {
    const text = line.trim();
    const note = text.startsWith("import ")
      ? "Import a dependency used by this example."
      : text.startsWith("class ")
        ? "Define a small, testable module."
        : text.startsWith("def ")
          ? "Make one operation explicit and reusable."
          : text.startsWith("assert ")
            ? "Check a numerical, shape, or data-split invariant."
            : text.startsWith("print(")
              ? "Display the observed result."
              : text.startsWith("#")
                ? text.replace(/^#+\s*/, "")
                : `Trace this step through ${shapes}.`;
    return `Line ${index + 1} · ${note}`;
  }),
});

const snippets: Snippet[] = [
  makeSnippet(
    "regularization-toy",
    "Compare dropout and BatchNorm train/eval behavior",
    "Verify stochastic masking in train mode, stable output in eval, and BatchNorm running-stat evaluation.",
    "x [B,8] → Linear → BatchNorm1d → ReLU → Dropout → [B,4]",
    `import torch
from torch import nn

torch.manual_seed(31)
block = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(8, 4))
x = torch.randn(16, 8)
block.train()
train_a, train_b = block(x), block(x)
assert train_a.shape == (16, 4) and not torch.equal(train_a, train_b)
block.eval()
eval_a, eval_b = block(x), block(x)
assert torch.equal(eval_a, eval_b)
assert block[1].num_batches_tracked.item() > 0
print(f"output={list(eval_a.shape)}; dropout varies in train and is disabled in eval")`,
    "output=[16, 4]; dropout varies in train and is disabled in eval",
  ),
  makeSnippet(
    "regularization-curves",
    "Record train and validation loss by epoch",
    "Create reproducible validation history and identify a checkpoint using validation only.",
    "epoch → train loss, validation loss, validation accuracy",
    `import torch

history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
for train_loss, validation_loss, validation_accuracy in [
    (1.8, 1.7, 0.42), (1.1, 1.2, 0.66), (0.6, 1.4, 0.61)
]:
    history["train_loss"].append(train_loss)
    history["validation_loss"].append(validation_loss)
    history["validation_accuracy"].append(validation_accuracy)
best_epoch = min(range(len(history["validation_loss"])), key=history["validation_loss"].__getitem__)
assert len(history["train_loss"]) == 3 and best_epoch == 1
assert history["train_loss"][-1] < history["train_loss"][0]
print(f"best validation checkpoint epoch={best_epoch + 1}; final train loss={history['train_loss'][-1]:.1f}")`,
    "best validation checkpoint epoch=2; final train loss=0.6",
  ),
  makeSnippet(
    "batchnorm-numerics",
    "Apply BatchNorm scale and shift numerically",
    "Use a known normalized value to see how learned gamma and beta change the output.",
    "x [B,F] → normalize → γx̂ + β → [B,F]",
    `import torch
from torch import nn

norm = nn.BatchNorm1d(2, affine=True, track_running_stats=True)
norm.eval()
with torch.no_grad():
    norm.running_mean.copy_(torch.tensor([1.0, 2.0]))
    norm.running_var.copy_(torch.tensor([4.0, 9.0]))
    norm.weight.copy_(torch.tensor([2.0, 0.5]))
    norm.bias.copy_(torch.tensor([1.0, -1.0]))
x = torch.tensor([[3.0, 5.0]])
y = norm(x)
expected = torch.tensor([[2.9999975, -0.4999997]])
assert torch.allclose(y, expected, atol=1e-5)
print(f"normalized and affine output={y.tolist()}")`,
    "normalized and affine output=[[3.0, -0.5]] (approximately)",
  ),
  makeSnippet(
    "weight-decay-update",
    "Observe an L2-style weight-decay update",
    "Isolate the shrink component for SGD so weight_decay and optimizer-step placement are concrete.",
    "w → optimizer.step(weight_decay) → smaller |w|",
    `import torch

parameter = torch.nn.Parameter(torch.tensor([2.0]))
optimizer = torch.optim.SGD([parameter], lr=0.1, weight_decay=0.25)
parameter.grad = torch.zeros_like(parameter)
optimizer.step()
assert torch.allclose(parameter, torch.tensor([1.95]))
assert parameter.abs().item() < 2.0
print(f"weight after shrink-only step={parameter.item():.2f}")`,
    "weight after shrink-only step=1.95",
  ),
  makeSnippet(
    "optimizer-comparison",
    "Compare SGD, momentum, Adam and AdamW setup",
    "Keep optimizer choice separate from learned parameter values and inspect the configurations.",
    "same parameter set → different update rules",
    `import torch

def make_optimizers(parameters):
    return {
        "SGD": torch.optim.SGD(parameters, lr=0.05),
        "SGD+momentum": torch.optim.SGD(parameters, lr=0.05, momentum=0.9),
        "Adam": torch.optim.Adam(parameters, lr=0.001),
        "AdamW": torch.optim.AdamW(parameters, lr=0.001, weight_decay=1e-4),
    }

parameter = torch.nn.Parameter(torch.ones(2))
optimizers = make_optimizers([parameter])
assert set(optimizers) == {"SGD", "SGD+momentum", "Adam", "AdamW"}
assert optimizers["SGD+momentum"].param_groups[0]["momentum"] == 0.9
assert optimizers["AdamW"].param_groups[0]["weight_decay"] == 1e-4
print("SGD, momentum, Adam and AdamW are configured")`,
    "SGD, momentum, Adam and AdamW are configured",
  ),
  makeSnippet(
    "regularized-ann",
    "Build a regularized ANN and run one update",
    "Compose Linear, BatchNorm, ReLU and Dropout in the hidden block with logits at the output.",
    "x [16,20] → hidden [16,32] → logits [16,10]",
    `import torch
from torch import nn

torch.manual_seed(5)
model = nn.Sequential(
    nn.Linear(20, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 10),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
images, labels = torch.randn(16, 20), torch.arange(16) % 10
model.train()
optimizer.zero_grad(set_to_none=True)
loss = nn.CrossEntropyLoss()(model(images), labels)
loss.backward()
optimizer.step()
model.eval()
with torch.inference_mode():
    logits = model(images)
assert logits.shape == (16, 10) and loss.isfinite()
print(f"loss={loss.item():.3f}, eval logits={list(logits.shape)}")`,
    "Finite scalar loss and eval logits [16,10].",
  ),
  makeSnippet(
    "optuna-synthetic",
    "Run a tiny reproducible Optuna study",
    "Demonstrate suggest_int, suggest_float, categorical choices, a seeded sampler and study reporting on a synthetic objective.",
    "trial params → deterministic validation-like score → study trials",
    `"""Requires optional Optuna: python -m pip install optuna"""
import torch
import optuna

def objective(trial):
    width = trial.suggest_int("width", 4, 16, step=4)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    optimizer_bonus = 0.05 if optimizer_name == "Adam" else 0.0
    score = 1.0 - ((width - 12) / 16) ** 2 - abs(learning_rate - 0.01) + optimizer_bonus
    return score

sampler = optuna.samplers.TPESampler(seed=23)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=8)
assert len(study.trials) == 8
assert set(study.best_params) == {"width", "learning_rate", "optimizer"}
print(f"trials={len(study.trials)}, best score={study.best_value:.3f}")`,
    "trials=8 with a reproducible best trial; the sampled optimum is not guaranteed globally optimal.",
    "Python 3.10+ · optional Optuna · no datasets or downloads",
  ),
  makeSnippet(
    "optuna-ann-safe",
    "Tune on validation and touch test once",
    "Show an Optuna objective that trains only on train, returns validation loss, refits the selected setup, then evaluates the untouched test set.",
    "train → trial validation → best params refit → final test",
    `"""Requires optional Optuna: python -m pip install optuna"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

torch.manual_seed(41)
features = torch.randn(180, 8)
targets = (features[:, 0] + 0.4 * features[:, 1] > 0).long()
train = TensorDataset(features[:100], targets[:100])
valid = TensorDataset(features[100:140], targets[100:140])
test = TensorDataset(features[140:], targets[140:])  # Not passed into objective.

def fit_and_score(width, learning_rate, epochs, train_data, valid_data):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(train_data, batch_size=20, shuffle=True, generator=torch.Generator().manual_seed(9))
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
    model.eval()
    total_loss = count = 0.0
    with torch.inference_mode():
        for x, y in DataLoader(valid_data, batch_size=32):
            total_loss += nn.CrossEntropyLoss(reduction="sum")(model(x), y).item()
            count += y.numel()
    return total_loss / count, model

def objective(trial):
    width = trial.suggest_categorical("width", [8, 16])
    lr = trial.suggest_float("lr", 0.002, 0.03, log=True)
    loss, _ = fit_and_score(width, lr, 3, train, valid)
    return loss  # Validation only; never read test here.

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=2))
study.optimize(objective, n_trials=2)
best_loss, final_model = fit_and_score(study.best_params["width"], study.best_params["lr"], 3, train, valid)
final_model.eval()
correct = count = 0
with torch.inference_mode():
    for x, y in DataLoader(test, batch_size=32):
        correct += (final_model(x).argmax(dim=1) == y).sum().item()
        count += y.numel()
assert count == len(test) and 0 <= correct / count <= 1
print(f"best validation loss={best_loss:.3f}; one final test accuracy={correct / count:.3f}")`,
    "Two validation-only trials; best configuration refit; final test evaluated once over 40 examples.",
    "Python 3.10+ · PyTorch · optional Optuna · CPU · synthetic data",
  ),
  makeSnippet(
    "optuna-plotting",
    "Create optional Optuna diagnostic plots",
    "Show history, parallel-coordinate and importance plots only when the optional visualization dependencies are installed.",
    "completed study → plotly figures → local HTML display",
    `"""Requires optional Optuna visualization extras."""
import torch

def show_plots(study):
    try:
        import optuna
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
    except ImportError as error:
        raise RuntimeError("Install Optuna visualization extras to create plots") from error
    figures = [
        plot_optimization_history(study),
        plot_parallel_coordinate(study),
        plot_param_importances(study),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        figure.show()  # Local interactive output; no hosted tracking service is required.

print("Call show_plots(study) after optimization when Optuna visualization extras are available.")
assert callable(show_plots)`,
    "Defines local optional plotting for optimization history, parallel coordinates and parameter importance.",
    "Python 3.10+ · optional Optuna + Plotly · no hosted service",
  ),
  makeSnippet(
    "study-experiments-log",
    "Keep a reproducible experiment log",
    "Record split policy, seed, search definition, software and trial outcomes for an auditable study.",
    "run config + trial params/value/state → experiment record",
    `import json
import torch

record = {
    "seed": 23,
    "split": "train/validation/test = 100/40/40; test held until final report",
    "sampler": "TPE with seed 23",
    "objective": "minimize validation cross entropy",
    "trials": [
        {"number": 0, "params": {"width": 8, "lr": 0.01}, "value": 0.48, "state": "COMPLETE"},
        {"number": 1, "params": {"width": 16, "lr": 0.005}, "value": 0.41, "state": "COMPLETE"},
    ],
}
serialized = json.dumps(record, sort_keys=True)
restored = json.loads(serialized)
assert len(restored["trials"]) == 2
assert restored["split"].endswith("final report")
print(f"recorded {len(restored['trials'])} trials; config is JSON serializable")`,
    "recorded 2 trials; config is JSON serializable",
  ),
];

const byId = Object.fromEntries(snippets.map((snippet) => [snippet.id, snippet]));
const section = (id: string, title: string, paragraphs: string[]) => ({
  id,
  title,
  paragraphs,
});
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
}): Lesson => ({
  id: args.id,
  moduleId:
    args.id === "hyperparameter-search" ||
    args.id.startsWith("optuna-") ||
    args.id.startsWith("ann-") ||
    args.id === "dynamic-ann"
      ? "tuning"
      : "vision",
  title: args.title,
  subtitle: args.subtitle,
  minutes: 9,
  source: { start: args.source[0], end: args.source[1], kind: "source-derived" },
  objectives: [
    args.objective,
    "Read the tensor/parameter flow and verify it with the runnable example.",
    "Separate source observations from reproducible conclusions.",
  ],
  intuition: args.intuition,
  equation: args.equation,
  numerical: args.numerical,
  shapes: args.shapes,
  diagram: args.diagram,
  sections: args.sections,
  snippet: byId[args.snippet],
  template: byId[args.template],
  additionalSnippets: args.additional?.map((id) => byId[id]),
  pitfalls: args.pitfalls,
  production: args.production,
  revision: args.revision,
  questions: args.questions.map(([question, answer]) => ({ question, answer })),
});

const regularizationSections = [
  section("overfit", "Read the gap before choosing a fix", [
    "A training score that keeps improving while validation quality stalls or declines is evidence the fit is not transferring as well to held-out examples. Plot training and validation loss together; confirm the split, metric and class balance before acting.",
    "The source reports approximately 98% training and 88.7% test accuracy for one setup. It is a reported result, not a benchmark promise. Its seven intervention ideas include more data, a smaller model, regularization, dropout, augmentation, batch normalization and early stopping; choose based on the data and failure mode.",
  ]),
  section("split-protocol", "Validation guides choices; test stays untouched", [
    "Use training data to fit parameters, validation data to choose epoch count and settings, and the test set for one final report after the model-selection process. Repeatedly checking test accuracy during tuning turns it into part of the selection process.",
    "Learning curves show an experiment, not causality by themselves. Compare repeated seeds or controlled runs before attributing a change to one regularizer.",
  ]),
];

export const phase06Lessons: Lesson[] = [
  makeLesson({
    id: "overfitting",
    title: "Use validation curves to diagnose overfitting",
    subtitle: "Training fit and generalization answer different questions",
    source: [60, 61],
    diagram: {
      kind: "generalization",
      title: "Move through an illustrative train/validation curve",
      description:
        "The curves are illustrative: training loss can keep falling after validation loss bottoms out. Select a stopping point with validation, then use test once for the final report.",
    },
    objective:
      "Distinguish underfit, useful fit and an overfit gap using training and validation evidence.",
    intuition:
      "A student who memorizes the practice sheet may score well on familiar examples while struggling with new questions. Validation acts like a separate practice exam; test is the final held-out exam.",
    equation:
      "generalization gap = validation metric − training metric (state the metric and direction)",
    numerical:
      "A source run reports roughly 98% training accuracy and 88.7% test accuracy. The ten-point difference motivates investigation, but the values depend on its split and training setup.",
    shapes: [
      {
        name: "Training history",
        shape: "[epoch] loss/score",
        meaning: "Fit to examples used for updates.",
      },
      {
        name: "Validation history",
        shape: "[epoch] loss/score",
        meaning: "Held-out feedback used for choices.",
      },
      {
        name: "Test report",
        shape: "scalar(s) after selection",
        meaning: "Final estimate after choices are finished.",
      },
    ],
    sections: regularizationSections,
    snippet: "regularization-curves",
    template: "study-experiments-log",
    additional: ["regularized-ann"],
    pitfalls: [
      "Do not repeatedly pick settings by test score.",
      "Accuracy alone can hide class imbalance and unequal error costs.",
      "A gap is a diagnostic signal, not proof that one particular regularizer will help.",
      "Do not copy reported source performance as a guarantee.",
    ],
    production:
      "Store split identifiers, sample counts, seeds and metric definitions beside learning curves so changes can be reproduced.",
    revision: [
      "Training data updates model weights.",
      "Validation supports model and stopping choices.",
      "Test data is reserved until selection is complete.",
      "Compare both loss and task-relevant metrics across splits.",
    ],
    questions: [
      [
        "What pattern can suggest overfitting?",
        "Training loss falls while validation loss stops improving or begins rising.",
      ],
      [
        "Should test accuracy select an epoch?",
        "No. Choose with validation; reserve test for the final evaluation.",
      ],
      [
        "Does a ten-point score gap identify the cause?",
        "No. It motivates checks of splits, metrics, capacity, data and training before selecting an intervention.",
      ],
      [
        "Are source accuracy values guaranteed?",
        "No. They are observations from a specific reported run.",
      ],
    ],
  }),
  makeLesson({
    id: "dropout",
    title: "Dropout masks hidden activations during training",
    subtitle: "Randomly remove routes, then use the complete network at evaluation",
    source: [61, 66],
    diagram: {
      kind: "dropout",
      title: "Change the dropout probability and inspect an example mask",
      description:
        "The explorer applies inverted-dropout scaling to a fixed conceptual activation vector. A new training pass may sample a different mask; evaluation bypasses the mask.",
    },
    objective:
      "Explain dropout probability, hidden-layer placement, inverted scaling and train/eval behavior.",
    intuition:
      "If a team cannot rely on one person always doing the work, skills are distributed across more members. Dropout randomly removes some hidden activations on each training pass.",
    equation: "train: yᵢ = mᵢxᵢ/(1−p), mᵢ∼Bernoulli(1−p); eval: y=x",
    numerical:
      "With p=0.3, each hidden activation is dropped with probability 0.3; a retained activation is scaled by 1/0.7 so its expected value is preserved. The mask is sampled per forward pass.",
    shapes: [
      {
        name: "Hidden activations",
        shape: "[B,H]",
        meaning: "ReLU output for each sample and unit.",
      },
      {
        name: "Training mask",
        shape: "[B,H] Bernoulli",
        meaning: "Random zero/keep decisions.",
      },
      {
        name: "Evaluation output",
        shape: "[B,H]",
        meaning: "Dropout is identity in eval mode.",
      },
    ],
    sections: [
      section("placement", "Place dropout after hidden activation", [
        "A common block is Linear → BatchNorm → ReLU → Dropout, or Linear → ReLU → Dropout when BatchNorm is absent. Keep dropout in hidden layers; the classifier output should remain raw logits.",
        "PyTorch uses inverted dropout: retained activations scale during training. Calling model.eval() disables this random masking and restores deterministic full-network behavior. Evaluation still needs inference_mode() to avoid building a gradient graph.",
      ]),
      section("probability", "Interpret p as a probability, not a fixed count", [
        "nn.Dropout(p=0.3) independently zeros activations with probability 0.3. A batch does not necessarily lose exactly 30% of every hidden vector. Increase p cautiously: too much masking can make the model underfit.",
        "The analogy describes shared responsibility, while the implementation is a stochastic activation mask. Reproducibility requires setting seeds and recording the run configuration.",
      ]),
    ],
    snippet: "regularization-toy",
    template: "regularization-toy",
    additional: ["regularized-ann"],
    pitfalls: [
      "Do not expect identical outputs across training passes with dropout enabled.",
      "Do call eval() before deterministic evaluation.",
      "Do not remove inference_mode(); eval() alone still records gradients.",
      "Do not apply dropout to the class logits unless the objective specifically calls for it.",
    ],
    production:
      "Record dropout probability and seed. Check the train/eval transition in inference tests and evaluate calibration if probabilities drive decisions.",
    revision: [
      "Dropout is active in train mode and disabled in eval mode.",
      "p is each activation's drop probability.",
      "Inverted dropout scales retained training activations by 1/(1−p).",
      "Dropout usually sits in hidden layers after the activation.",
    ],
    questions: [
      [
        "What does Dropout(p=0.3) mean?",
        "Each activation has a 30% chance of being zeroed on a training forward pass.",
      ],
      [
        "Why scale retained activations during training?",
        "Inverted dropout keeps their expected activation equal to evaluation-time activations.",
      ],
      [
        "Does eval() disable gradient tracking?",
        "No. Use inference_mode() or no_grad() separately.",
      ],
      [
        "Why avoid dropout on the logits by default?",
        "The class scores should remain available to the loss without stochastic output masking.",
      ],
    ],
  }),
  makeLesson({
    id: "batch-normalization",
    title: "BatchNorm learns scale and shift while tracking running statistics",
    subtitle: "Normalize hidden features with explicit train/eval behavior",
    source: [62, 68],
    diagram: {
      kind: "batchnorm",
      title: "Adjust gamma and beta to transform normalized values",
      description:
        "For each feature, BatchNorm normalizes using batch statistics in training or running statistics in evaluation, then applies learnable γ and β.",
    },
    objective:
      "Compute BatchNorm1d normalization, affine scale/shift and running-stat behavior, including small-batch caveats.",
    intuition:
      "A quality-control station recenters and rescales each feature so later layers receive a more controlled distribution, while learned gamma and beta retain flexibility.",
    equation: "x̂=(x−μ)/√(σ²+ε); y=γx̂+β",
    numerical:
      "For x=3, running mean μ=1, variance σ²=4 and ε=10⁻⁵, x̂≈1. If γ=2 and β=1, output y≈3.",
    shapes: [
      { name: "Linear output", shape: "[B,F]", meaning: "F features per sample." },
      {
        name: "Batch statistics",
        shape: "[F] μ, σ²",
        meaning: "Feature statistics over the training minibatch.",
      },
      {
        name: "Affine output",
        shape: "[B,F]",
        meaning: "Normalized features scaled by γ and shifted by β.",
      },
    ],
    sections: [
      section("math", "Normalize, then learn an affine transform", [
        "For each feature j, BatchNorm computes a centered and variance-scaled value using a small ε for numerical stability. Learnable γⱼ and βⱼ then scale and shift it: yⱼ=γⱼx̂ⱼ+βⱼ.",
        "In an MLP, nn.BatchNorm1d(F) expects F features, typically shaped [B,F]. Place it after Linear and before ReLU: Linear → BatchNorm → ReLU. Its affine parameters are learned model weights; running mean/variance are buffers updated during training.",
      ]),
      section("modes", "Train mode uses batch statistics; eval uses running statistics", [
        "With track_running_stats=True, training updates running statistics and uses the current batch for normalization. Evaluation uses stored running values and does not update them. This makes predictions independent of other test examples in the current batch.",
        "Very small training batches produce noisy statistics; a final batch of size one can fail for BatchNorm1d in training mode. Use drop_last when suitable, alter batching, or choose a normalization method suited to the data. The source describes BatchNorm as fixing internal covariate shift; that is an intuitive historical explanation, not a complete modern account of why it helps.",
      ]),
    ],
    snippet: "batchnorm-numerics",
    template: "regularization-toy",
    additional: ["regularized-ann"],
    pitfalls: [
      "Set num_features to the preceding layer's output width.",
      "Do not forget eval() before evaluation; it changes BatchNorm statistics behavior.",
      "Check for tiny or singleton final batches in training.",
      "Avoid claiming BatchNorm always speeds up convergence or prevents overfit.",
    ],
    production:
      "Keep learned affine parameters and running buffers in state_dict. Test inference after loading the checkpoint in eval mode and with representative batch sizes.",
    revision: [
      "BatchNorm1d(F) normalizes F features.",
      "It learns gamma and beta.",
      "Train mode updates/uses minibatch statistics; eval uses running statistics.",
      "Small batches can make estimates noisy or invalid.",
    ],
    questions: [
      [
        "What is epsilon for?",
        "It stabilizes division by the standard deviation when variance is very small.",
      ],
      [
        "What do gamma and beta do?",
        "They are learned per-feature scale and shift after normalization.",
      ],
      [
        "Which statistics does BatchNorm use in eval mode?",
        "Running mean and variance accumulated during training, when running-stat tracking is enabled.",
      ],
      [
        "What issue can batch size one cause?",
        "BatchNorm1d cannot estimate training batch variance from a singleton batch in common training configurations.",
      ],
    ],
  }),
  makeLesson({
    id: "weight-decay",
    title: "Weight decay discourages unnecessarily large weights",
    subtitle: "Regularize through the optimizer update",
    source: [63, 70],
    diagram: {
      kind: "weight-decay",
      title: "Change lambda to see a shrink-only SGD step",
      description:
        "This one-dimensional illustration isolates decay with zero task gradient. Real optimizer updates also include the loss gradient; paths and outcomes depend on the optimizer.",
    },
    objective:
      "Relate an L2 penalty to optimizer weight decay and distinguish coupled L2 from AdamW's decoupled decay.",
    intuition:
      "A budget constraint makes very large coefficients costly, encouraging the model to explain the data with less extreme parameter values.",
    equation:
      "L_total=L_data+λ∑w²; SGD coupled gradient includes 2λw (convention-dependent factor)",
    numerical:
      "For SGD with lr=0.1, decay=0.25, parameter w=2 and zero task gradient, a shrink-only step gives w′=2−0.1×0.25×2=1.95 under the common coupled-SGD update.",
    shapes: [
      {
        name: "Parameters",
        shape: "many scalar weights",
        meaning: "Optimizer-owned model parameters.",
      },
      {
        name: "Penalty",
        shape: "scalar",
        meaning: "Adds a size-dependent term to the objective under coupled L2.",
      },
      {
        name: "Updated weights",
        shape: "same parameter shapes",
        meaning: "Task gradient and decay both affect the step.",
      },
    ],
    sections: [
      section("penalty", "Understand the size penalty", [
        "An L2 objective adds a coefficient times the sum of squared weights to the data loss (some definitions include one-half, changing the factor in the derivative). Larger weights contribute more strongly because they are squared.",
        "In SGD, weight_decay commonly implements coupled L2 by adding a parameter-proportional term to the gradient. The source's λ=1e-4 is one reported setting; tune it on validation. Too much regularization can underfit.",
      ]),
      section("adamw", "AdamW is decoupled weight decay", [
        "Adaptive optimizers scale gradients using running moments. Applying L2 inside that gradient is not mathematically identical to decaying weights separately. AdamW applies a decoupled shrink update; its weight_decay parameter should not be described as exactly the same objective penalty as coupled L2 for Adam.",
        "The regularization strength, optimizer, learning rate and parameter groups interact. It can be useful to exclude bias or normalization parameters from decay, but that is an engineering choice to validate rather than a universal rule.",
      ]),
    ],
    snippet: "weight-decay-update",
    template: "optimizer-comparison",
    additional: ["regularized-ann"],
    pitfalls: [
      "Weight decay is an optimizer setting, not a neural network layer.",
      "State whether the L2 objective includes a one-half factor when writing derivatives.",
      "AdamW decoupled decay is not identical to coupled L2 in adaptive optimizers.",
      "Very large decay can cause underfitting.",
    ],
    production:
      "Record optimizer, parameter-group decay values, learning rate and checkpoint together. Validate any no-decay parameter grouping on the target task.",
    revision: [
      "L2 penalizes squared parameter magnitude.",
      "SGD weight decay contributes a shrink term to its update.",
      "AdamW decouples weight decay from its adaptive gradient calculation.",
      "Tune regularization using validation, not test results.",
    ],
    questions: [
      [
        "Why does squaring a weight penalize large values more?",
        "The penalty grows quadratically with magnitude.",
      ],
      [
        "Where is weight_decay configured in the example?",
        "On the optimizer or an optimizer parameter group.",
      ],
      [
        "Is Adam weight decay always equivalent to AdamW?",
        "No. Coupled L2 and decoupled AdamW updates differ under adaptive scaling.",
      ],
      [
        "What may happen if decay is too strong?",
        "The model can underfit because parameter updates are constrained too much.",
      ],
    ],
  }),
  makeLesson({
    id: "regularization-comparison",
    title: "Combine regularizers in the right places",
    subtitle: "A hidden block, optimizer choice and readable cheat sheet",
    source: [70, 71],
    diagram: {
      kind: "optimizer-landscape",
      title: "Compare illustrative optimizer paths on a loss surface",
      description:
        "The contours and paths are schematic, not benchmark results. Momentum uses a velocity; adaptive methods scale updates; AdamW separates decay.",
    },
    objective:
      "Place BatchNorm and dropout in a hidden block, select an optimizer deliberately and interpret the resulting experiment as a controlled comparison.",
    intuition:
      "Dropout changes which hidden routes are active, BatchNorm controls feature scale, weight decay affects parameter updates, and the optimizer determines how gradients move weights.",
    equation:
      "Linear → BatchNorm → ReLU → Dropout → Linear(logits); optimizer.step() updates parameters",
    numerical:
      "A 20→32→10 classifier has hidden activations [B,32] and logits [B,10]. The hidden block carries 1,024 affine BatchNorm values (32 gamma plus 32 beta; BatchNorm buffers are not trainable parameters).",
    shapes: [
      { name: "Input", shape: "[B,20]", meaning: "Synthetic or tabular features." },
      {
        name: "Regularized hidden",
        shape: "[B,32]",
        meaning: "Linear, BatchNorm, ReLU and Dropout.",
      },
      { name: "Output", shape: "[B,10]", meaning: "Raw logits for cross entropy." },
    ],
    sections: [
      section("block", "Keep normalization, activation and dropout order visible", [
        "The source's combined hidden block uses Linear → BatchNorm1d → ReLU → Dropout, followed by an output Linear layer. BatchNorm and Dropout belong in hidden computation; do not normalize or randomly mask the classifier scores by default.",
        "The code example makes one update with AdamW. It is a shape and mode smoke test, not evidence that this exact combination outperforms alternatives.",
      ]),
      section("optimizers", "Choose SGD, momentum, Adam or AdamW with evidence", [
        "SGD follows the current gradient. Momentum accumulates a velocity that can smooth updates and preserve direction. Adam adapts per-parameter step sizes with first- and second-moment estimates. AdamW adds decoupled weight decay. Learning rate and decay ranges should suit each optimizer.",
        "No optimizer is universally best. Compare under a fixed split and budget, log seeds and validation curves, and keep schedulers as a separate decision. A scheduler modifies the learning rate over training; it does not replace the optimizer.",
      ]),
    ],
    snippet: "regularized-ann",
    template: "optimizer-comparison",
    additional: ["regularization-toy", "batchnorm-numerics", "weight-decay-update"],
    pitfalls: [
      "BatchNorm goes before ReLU in this common block; dropout follows the activation.",
      "Do not compare optimizers with different data splits or unreported budgets.",
      "Do not equate batch-normalization running buffers with trainable parameters.",
      "Do not present schematic optimizer trajectories as measured performance.",
    ],
    production:
      "Log model configuration, parameter groups, optimizer and scheduler settings, seed, split identity, validation metrics and checkpoint selection rule.",
    revision: [
      "Hidden block: Linear → BatchNorm → ReLU → Dropout.",
      "Final layer returns raw logits.",
      "Optimizer and scheduler have distinct roles.",
      "Run a controlled validation-based comparison before choosing.",
    ],
    questions: [
      [
        "Where does BatchNorm appear in the shown hidden block?",
        "After Linear and before ReLU.",
      ],
      ["Where is dropout placed?", "After ReLU in the hidden block."],
      [
        "How does momentum differ from plain SGD?",
        "It keeps a velocity from past gradients to smooth or reinforce update directions.",
      ],
      [
        "Does AdamW's name mean it is just Adam plus an L2 loss?",
        "No. AdamW decouples weight decay from the adaptive gradient update.",
      ],
    ],
  }),
  makeLesson({
    id: "hyperparameter-search",
    title: "Search a bounded space instead of guessing settings",
    subtitle: "Grid, random and model-based trial selection",
    source: [72, 73],
    diagram: {
      kind: "search-space",
      title: "Choose regions and inspect previous trial outcomes",
      description:
        "A small illustrative search map shows grid coverage, random samples and a guided TPE proposal. TPE uses observed trial history but does not guarantee a global optimum.",
    },
    objective:
      "Compare grid, random and model-based search while keeping objective, budget and validation protocol explicit.",
    intuition:
      "Grid search checks every point on a small menu; random search samples the menu; a model-based sampler uses prior trial outcomes to choose new proposals.",
    equation:
      "trial t: θₜ ~ sampler(history of earlier trials); objective returns validation metric",
    numerical:
      "A 4×4 grid checks 16 combinations. Random search with 16 trials checks 16 sampled combinations; neither is guaranteed to find the best point outside its budget or sampled locations.",
    shapes: [
      {
        name: "Search domain",
        shape: "ranges/categories",
        meaning: "Candidate hyperparameters, not learned tensors.",
      },
      {
        name: "Trial history",
        shape: "[T, params, value]",
        meaning: "Previous configurations and objective outcomes.",
      },
      {
        name: "Best observed",
        shape: "params + value",
        meaning: "Best trial found within the completed budget.",
      },
    ],
    sections: [
      section("methods", "Match search strategy to the space and budget", [
        "Grid search exhausts a discrete product and becomes expensive as dimensions grow. Random search covers a bounded budget and can be effective when only a few dimensions matter. Bayesian or model-based approaches use prior results to guide proposals; Optuna's TPE is one such sampler.",
        "TPE models distributions of promising and less-promising parameter regions and samples candidates using that history. It is a heuristic sampler, not an exact solver and not a guarantee of a global optimum. The source's claim that it quickly discovers a globally best area is too strong.",
      ]),
      section("contract", "Make one trial return one well-defined metric", [
        "The objective receives a trial, requests values, creates and trains a configuration, evaluates validation data, and returns the metric matching study direction. For accuracy maximize; for loss minimize. State the data splits, resource budget, seed and stopping rule.",
        "Search cost is approximately trials × per-trial fit cost. Start with a cheap synthetic objective to understand the API, then constrain real model trials to a budget that fits available hardware.",
      ]),
    ],
    snippet: "optuna-synthetic",
    template: "optuna-synthetic",
    additional: ["study-experiments-log"],
    pitfalls: [
      "A larger search space or budget increases compute cost.",
      "A sampler uses trial outcomes but cannot promise the global optimum.",
      "Keep study direction aligned with the returned metric.",
      "Do not let the test set drive trial selection.",
    ],
    production:
      "Persist study storage and record sampler seed, package versions, dataset split IDs, objective code version and stopping criteria.",
    revision: [
      "Hyperparameters are settings; weights are learned parameters.",
      "Grid, random and TPE choose trial configurations differently.",
      "One objective call returns one score/value.",
      "Report the best observed trial and its search budget.",
    ],
    questions: [
      [
        "What is the main limitation of a dense grid in many dimensions?",
        "The number of combinations grows multiplicatively and becomes expensive.",
      ],
      [
        "Does TPE guarantee finding a global optimum?",
        "No. It proposes trials based on an approximate model and finite history.",
      ],
      [
        "What does study direction control?",
        "Whether Optuna seeks to maximize or minimize the objective value.",
      ],
      [
        "How should compute cost be estimated?",
        "By number of trials multiplied by the cost of training and validating each trial.",
      ],
    ],
  }),
  makeLesson({
    id: "optuna-basics",
    title: "Optuna organizes optimization into studies and trials",
    subtitle: "A trial suggests parameters; an objective returns a value",
    source: [73, 75],
    diagram: {
      kind: "trial-comparison",
      title: "Compare trial parameters and objective values",
      description:
        "Each row is one completed trial. Color encodes a schematic objective value; study.best_trial is the best observed result for the chosen direction.",
    },
    objective:
      "Define study, trial, suggested parameter, objective and sampler and connect them in a minimal workflow.",
    intuition:
      "A study is an experiment notebook; each trial is one recipe. The objective cooks the recipe, measures it, and returns one score.",
    equation: "study.optimize(objective, n_trials=T); objective(trial) → float",
    numerical:
      "With direction='minimize', values 0.42, 0.31 and 0.36 make trial 1 the current best. This means best among these three observed trials.",
    shapes: [
      {
        name: "Study",
        shape: "collection of trials",
        meaning: "Optimization run and storage handle.",
      },
      {
        name: "Trial parameters",
        shape: "mapping[str,value]",
        meaning: "Suggested configuration for one run.",
      },
      {
        name: "Objective",
        shape: "scalar value",
        meaning: "Comparable metric with declared direction.",
      },
    ],
    sections: [
      section("terms", "Give each Optuna object one role", [
        "A study contains trials and has an optimization direction. A trial represents one configuration and offers suggest_int, suggest_float or suggest_categorical. The objective uses those suggestions to train/evaluate a model and returns a scalar. A sampler chooses proposals based on its algorithm and available history.",
        "Set a sampler explicitly when reproducibility matters; seed samplers that support seeds and seed model initialization and data shuffling too. A sampler seed alone does not make every accelerator kernel deterministic.",
      ]),
      section("study", "Create, optimize and inspect", [
        "Use create_study(direction='minimize' or 'maximize'), call study.optimize(objective, n_trials=...), then inspect best_value, best_params and trials_dataframe(). For persistent studies, use a database storage URL and a stable study name; SQLite is a local option for a single machine.",
        "Pruners can stop unpromising trials when the objective reports intermediate values. They are separate from samplers. Callbacks can record progress or stop a run based on a condition.",
      ]),
    ],
    snippet: "optuna-synthetic",
    template: "optuna-synthetic",
    additional: ["study-experiments-log"],
    pitfalls: [
      "A study direction must match the metric orientation.",
      "A trial value is not comparable if splits or metric definitions change across trials.",
      "Do not confuse a pruner (stopping) with a sampler (proposing).",
      "Do not promise cross-device determinism from one seed.",
    ],
    production:
      "Use named, versioned studies and persistent storage for resumable runs. Store code/data version alongside trial values.",
    revision: [
      "Study groups trials.",
      "Trial exposes suggested hyperparameters.",
      "Objective trains/evaluates and returns a scalar.",
      "Sampler proposes; pruner may stop a trial early.",
    ],
    questions: [
      [
        "What does a trial contain?",
        "A single proposed parameter configuration, its objective result and execution state.",
      ],
      [
        "What does the objective return?",
        "A scalar metric that Optuna minimizes or maximizes.",
      ],
      ["What is the sampler's role?", "It proposes parameter values for new trials."],
      [
        "What does a pruner do?",
        "It can stop a trial early based on intermediate results.",
      ],
    ],
  }),
  makeLesson({
    id: "optuna-tabular",
    title: "Prepare tabular data before tuning its classifier",
    subtitle: "Preprocessing and cross-validation stay inside the training boundary",
    source: [74, 75],
    diagram: {
      kind: "search-space",
      title: "Trial models see training folds, not the final test set",
      description:
        "Preprocessing is fit within each training fold. Validation/search evaluates candidate settings; the held-out test set remains outside objective calls.",
    },
    objective:
      "Explain the source's tabular Optuna example while correcting preprocessing leakage and unreliable cross-validation placement.",
    intuition:
      "If a study has already peeked at the final exam while choosing settings, the final score is no longer an independent report.",
    equation:
      "search objective = mean CV score on train partition; final test after best configuration is selected",
    numerical:
      "Three-fold CV on 300 training records fits three models per trial. Twenty trials therefore require 60 fold fits, before final refit.",
    shapes: [
      {
        name: "Training split",
        shape: "[Ntrain,F]",
        meaning: "Used for fold fitting and trial comparison.",
      },
      {
        name: "CV result",
        shape: "3 fold scores → scalar",
        meaning: "Mean fold score returned by the objective.",
      },
      {
        name: "Test split",
        shape: "[Ntest,F]",
        meaning: "Held outside tuning and read at final evaluation.",
      },
    ],
    sections: [
      section("preprocess", "Fit transforms inside each training fold", [
        "The source describes the Pima diabetes dataset, replacing invalid zeros, imputation, scaling and a random-forest objective using cross-validation. Treat the dataset and code as an example; the exported snippet is clipped and should be reconstructed.",
        "For leakage-safe cross-validation, use a Pipeline that contains imputation, scaling when needed, and the estimator. cross_val_score clones and fits that pipeline separately on each training fold. Do not impute or scale the entire dataset before folds are created.",
      ]),
      section("final-test", "Use one final held-out evaluation", [
        "Split the original records into development and final-test partitions before tuning. Pass only development data to the objective. Once search and any validation-based decisions are complete, refit with selected settings on the allowed development data and evaluate the final test set once.",
        "For high-stakes or small datasets, choose stratification, nested CV or repeated CV based on the evaluation question. These are supplemental protocol choices beyond the source's short demonstration.",
      ]),
    ],
    snippet: "optuna-ann-safe",
    template: "optuna-ann-safe",
    additional: ["optuna-synthetic"],
    pitfalls: [
      "Never run cross-validation on records that were already used to fit preprocessing globally.",
      "Do not return test accuracy from a trial objective.",
      "Report fold design and variation, not only a rounded mean.",
      "Keep validation and final test roles explicit.",
    ],
    production:
      "Log fold identifiers, preprocessing pipeline, scoring definition, random states and the exact final refit procedure.",
    revision: [
      "A CV objective scores only development data.",
      "Fit preprocessing within each fold.",
      "Keep final test outside the study objective.",
      "Account for folds × trials in compute planning.",
    ],
    questions: [
      [
        "Why use a preprocessing Pipeline in CV?",
        "It fits transformations on each fold's training partition and avoids validation-fold leakage.",
      ],
      [
        "How many fits for 20 trials and 3 folds?",
        "Sixty fold fits, plus any final refit.",
      ],
      [
        "Can the objective return test accuracy?",
        "No; return a validation or cross-validation metric and keep final test out of selection.",
      ],
      [
        "What should be reported for CV?",
        "Fold structure, metric, mean and useful variation/context.",
      ],
    ],
  }),
  makeLesson({
    id: "optuna-samplers",
    title: "Swap samplers without changing the objective",
    subtitle: "Random, grid and TPE have distinct proposal behavior",
    source: [75, 76],
    diagram: {
      kind: "search-space",
      title: "Compare grid, random and TPE proposals",
      description:
        "Grid enumerates declared combinations, random samples them, and TPE uses completed trials to shape proposals. The same objective can be passed to each study.",
    },
    objective:
      "Choose RandomSampler, GridSampler or TPESampler and account for reproducible seeds and categorical dimensions.",
    intuition:
      "A sampler is the route planner. The objective is the destination test; changing the route need not change what is measured.",
    equation: "sampler(search_space, history) → next configuration",
    numerical:
      "Four estimator counts × four depths define 16 GridSampler combinations. A random sampler can revisit a combination; a grid enumerates its finite product.",
    shapes: [
      {
        name: "Grid",
        shape: "Cartesian product",
        meaning: "All combinations in a discrete declared space.",
      },
      {
        name: "Random",
        shape: "sampled config per trial",
        meaning: "Independent proposals from configured distributions.",
      },
      {
        name: "TPE",
        shape: "history-guided proposal",
        meaning: "Uses completed trial distributions to guide new samples.",
      },
    ],
    sections: [
      section("selection", "Keep the objective fixed while comparing samplers", [
        "Optuna can use a seeded RandomSampler, a GridSampler with explicit categorical lists, or a TPESampler. Set the same objective, data split, metric and trial budget when comparing proposal strategies.",
        "Grid search is only appropriate for a small explicit set; a float range is not a finite grid until discretized. Random search can be a strong baseline. TPE can adapt proposal distributions as trials accumulate, but early suggestions may still be poor.",
      ]),
      section("reproducibility", "Seed the sampler and the model separately", [
        "Pass a seed to samplers that support one, seed model initialization and DataLoader generators, and record library versions. Parallel trial scheduling can change which completed trials are available when a new proposal is made, so exact sequence reproducibility may differ from sequential runs.",
        "A study direction and objective do not need to change when the sampler changes. This separation makes controlled comparisons easier to interpret.",
      ]),
    ],
    snippet: "optuna-synthetic",
    template: "optuna-synthetic",
    additional: ["study-experiments-log"],
    pitfalls: [
      "A finite grid requires explicit discrete values.",
      "Different samplers need equal budgets for a fair basic comparison.",
      "A sampler seed does not seed model randomness.",
      "Parallel optimization can affect proposal order and exact reproducibility.",
    ],
    production:
      "Record sampler class, sampler seed, trial count, parallelism and search-space definition with study metadata.",
    revision: [
      "Random and TPE are samplers, not objective functions.",
      "GridSampler requires a discrete search space.",
      "The same objective can be used with several samplers.",
      "Seed sampler, model and data shuffling separately.",
    ],
    questions: [
      ["How many settings are in a 4×4 grid?", "Sixteen combinations."],
      [
        "What does TPE use from history?",
        "Observed parameter configurations and objective values to guide future proposals.",
      ],
      [
        "Does changing sampler require rewriting the objective?",
        "Usually not; create the study with another sampler and keep the objective fixed.",
      ],
      [
        "Why can parallel runs be harder to reproduce exactly?",
        "The set and order of completed trials visible to a sampler can vary with scheduling.",
      ],
    ],
  }),
  makeLesson({
    id: "optuna-plots",
    title: "Read optimization history and parameter views",
    subtitle: "Plots diagnose a study; they do not validate the study design",
    source: [76, 77],
    diagram: {
      kind: "trial-comparison",
      title: "Compare objective by trial and parameter",
      description:
        "The local explorer shows trial values and parameter choices. Real Optuna history, parallel coordinate, slice, contour and importance figures need optional Plotly dependencies.",
    },
    objective:
      "Interpret optimization history, parallel coordinates, slice/contour views and parameter importance with their limits.",
    intuition:
      "Each plot is a different lens on the same experiment log: progress over time, combinations across trials, and possible parameter relationships.",
    equation:
      "plot = visualization(completed trial records), not an independent performance estimate",
    numerical:
      "With 20 trials, optimization history has at most 20 objective points; a parallel-coordinate line links one trial's parameter values to its objective.",
    shapes: [
      {
        name: "History plot",
        shape: "trial number × value",
        meaning: "Shows observations and incumbent best.",
      },
      {
        name: "Parallel coordinates",
        shape: "one polyline per trial",
        meaning: "Connects parameter values to objective.",
      },
      {
        name: "Importance",
        shape: "parameter → estimated contribution",
        meaning: "Sampler/model-based importance estimate for this study.",
      },
    ],
    sections: [
      section("views", "Use complementary views to ask concrete questions", [
        "Optimization history shows values by trial and can show the best-so-far trajectory. Parallel coordinates connect parameter values for each trial to the objective. Slice plots show one parameter against the objective; contour views show pairwise interactions. Parameter-importance plots estimate which searched values explain variation in the study.",
        "These views are observational summaries of the trials and search procedure. Sparse trials, correlated parameters and conditional spaces can make visual patterns misleading. An importance percentage is not a causal effect.",
      ]),
      section("optional", "Keep plotting optional and local", [
        "Optuna's visualization module uses optional plotting dependencies. Save figures as local HTML or display in a local notebook; do not require a hosted account or remote plotting API for the handbook.",
        "The example separates plotting from training. A completed study can be saved and analyzed later, and plots should include the study direction and metric context.",
      ]),
    ],
    snippet: "optuna-plotting",
    template: "optuna-plotting",
    additional: ["study-experiments-log"],
    pitfalls: [
      "Do not infer causality from a parameter-importance chart.",
      "Plots cannot repair test leakage or a biased split.",
      "Sparse histories may not support stable importance estimates.",
      "Optional visualization packages should not be required for core training.",
    ],
    production:
      "Archive trial data and plotting code with the study. Include metric, direction and search-space metadata in report captions.",
    revision: [
      "Optimization history follows objective values over trials.",
      "Parallel coordinates connect parameter choices to outcomes.",
      "Importance is model-based/estimated, not causal proof.",
      "Visualization dependencies can remain optional.",
    ],
    questions: [
      [
        "What does optimization history display?",
        "Trial objective values and often the best-so-far value.",
      ],
      [
        "What does each parallel-coordinate line represent?",
        "One trial across its parameters and objective.",
      ],
      [
        "Does estimated parameter importance prove causality?",
        "No. It summarizes a particular finite study and search space.",
      ],
      [
        "Why keep plotting dependencies optional?",
        "Core optimization remains usable without visualization extras or hosted services.",
      ],
    ],
  }),
  makeLesson({
    id: "dynamic-search",
    title: "Build conditional search spaces with define-by-run",
    subtitle: "Suggest only parameters relevant to the chosen model family",
    source: [77, 79],
    diagram: {
      kind: "search-space",
      title: "Branch after choosing a model family",
      description:
        "A categorical model choice opens a family-specific branch: SVM, random forest or boosting use different parameters. Each branch returns the same validation objective.",
    },
    objective:
      "Use categorical model selection with conditional hyperparameters and explain study tracking, storage and compute boundaries.",
    intuition:
      "A recipe changes by model family: an SVM needs kernel and C; a forest needs tree counts and depth. A conditional space does not ask irrelevant questions.",
    equation:
      "model = suggest_categorical(...); params = suggest_* only inside chosen branch",
    numerical:
      "A study may have eight trials across three families; the `C` parameter exists only on SVM trials, so it is absent for forest trials.",
    shapes: [
      { name: "Family choice", shape: "categorical", meaning: "Selects a model branch." },
      {
        name: "Conditional params",
        shape: "family-specific mapping",
        meaning: "Only parameters for the chosen branch.",
      },
      {
        name: "Study record",
        shape: "trial states/values/params",
        meaning: "Supports resume and post-run analysis.",
      },
    ],
    sections: [
      section("conditional", "Define parameters as the trial is built", [
        "Optuna's define-by-run API lets the objective request a classifier name, then request only the values that classifier uses. SVM can branch over C, kernel and gamma; a forest can branch over tree count, depth and split sizes; boosting can add a learning rate.",
        "Conditional parameters make plots and comparisons more complex: one parameter applies only to a subset of trials. Compare model families on the same validation protocol and report their budgets.",
      ]),
      section("tracking", "Persist trials without requiring a tracking server", [
        "A study can store trials in memory or in a database-backed storage URL such as local SQLite. Named persistent studies can resume. trials_dataframe() supports tabular inspection, and callbacks can log progress or stop a study.",
        "Optuna can be integrated with external experiment trackers such as MLflow, but that is optional supplemental engineering. The reader and examples do not require a server, account or API key.",
      ]),
    ],
    snippet: "optuna-synthetic",
    template: "study-experiments-log",
    additional: ["optuna-plotting"],
    pitfalls: [
      "Do not compare family branches on different splits or scoring definitions.",
      "Conditional parameter importance applies to the trials where that parameter exists.",
      "Persistent storage requires a stable study name and compatible schema/version assumptions.",
      "MLflow integration is optional and not part of the core example.",
    ],
    production:
      "Persist the study name, storage URI, code version, data fingerprint and model-family branch definitions. Avoid placing credentials in source or study logs.",
    revision: [
      "Define-by-run requests parameters conditionally inside the objective.",
      "Different model families have different search spaces.",
      "SQLite can persist a local study; memory storage is temporary.",
      "External tracking is optional.",
    ],
    questions: [
      [
        "Why use conditional suggestions?",
        "They keep each trial's parameter set relevant to its selected model family.",
      ],
      [
        "Will every trial have a `C` parameter?",
        "No; it exists only for branches such as SVM that use it.",
      ],
      [
        "What is a benefit of persistent storage?",
        "A named study can be inspected and resumed across runs.",
      ],
      [
        "Is MLflow required for Optuna?",
        "No. It is an optional integration for experiment tracking.",
      ],
    ],
  }),
  makeLesson({
    id: "optuna-tracking",
    title: "Make studies reproducible and safe to resume",
    subtitle: "Storage, callbacks, pruners and experiment metadata",
    source: [79, 79],
    diagram: {
      kind: "trial-comparison",
      title: "Follow a trial from suggestion to saved record",
      description:
        "A completed study records parameters, value and state. Storage enables later inspection; callbacks and pruners observe execution without changing the held-out test protocol.",
    },
    objective:
      "Record a reproducible study and distinguish storage, callbacks, pruning and external tracking.",
    intuition:
      "A lab notebook should preserve how each configuration was chosen and what data and code produced its score, not just the winning number.",
    equation:
      "experiment record = data/split + code/config + sampler/seed + trial history + final report",
    numerical:
      "A 50-trial study with 7 settings logs 350 parameter values at most, plus objective values and trial states; final model evaluation remains a separate record.",
    shapes: [
      {
        name: "Study storage",
        shape: "persistent trial table",
        meaning: "Stores configs, values and states.",
      },
      {
        name: "Callback",
        shape: "event → action",
        meaning: "Observes progress or requests stop/logging.",
      },
      {
        name: "Final report",
        shape: "chosen model + held-out metric",
        meaning: "Reported only after selection and refit.",
      },
    ],
    sections: [
      section("storage", "Separate saved trials from external tracking", [
        "Optuna storage persists study and trial records; a local SQLite database can be sufficient for a single-machine exercise. Use stable study names and avoid concurrent schema/version surprises when resuming older studies.",
        "A trials DataFrame helps audit parameters, objective values and states. Save the search-space definition, code revision, data split identity, seeds, package versions and hardware context as experiment metadata.",
      ]),
      section(
        "callbacks",
        "Callbacks observe progress; pruners stop unpromising trials",
        [
          "A callback receives study and trial events and can log progress or stop a study based on a policy. A pruner can stop an individual trial when intermediate values suggest low promise; the objective must report intermediate metrics for many pruning strategies.",
          "Neither callback nor storage chooses the metric protocol. Keep trial decisions on validation data. MLflow or other trackers can complement Optuna as optional supplemental tools.",
        ],
      ),
    ],
    snippet: "study-experiments-log",
    template: "study-experiments-log",
    additional: ["optuna-plotting"],
    pitfalls: [
      "In-memory studies disappear when the process exits.",
      "Store run context rather than relying on the best_params field alone.",
      "Pruning needs suitable intermediate reports and can bias comparisons if applied inconsistently.",
      "Do not store secrets or sensitive raw records in trial attributes.",
    ],
    production:
      "Use access-controlled storage, documented retention and sanitized metadata. Back up persistent study databases before schema upgrades or migration.",
    revision: [
      "Storage persists study/trial state.",
      "Callbacks observe or control study-level events.",
      "Pruners stop trials using intermediate values.",
      "Metadata is required to reproduce a winning run.",
    ],
    questions: [
      [
        "What does persistent storage preserve?",
        "Study and trial configurations, values, states and related metadata.",
      ],
      [
        "How does a callback differ from a sampler?",
        "A callback responds to study events; a sampler proposes parameter values.",
      ],
      [
        "What does a pruner use?",
        "Reported intermediate metrics or steps to decide whether a trial should stop.",
      ],
      [
        "What information should accompany best_params?",
        "Code/data versions, split, seed, sampler, search space, metric and resource budget.",
      ],
    ],
  }),
  makeLesson({
    id: "ann-search-space",
    title: "Budget the eight ANN hyperparameters",
    subtitle: "Architecture and training choices form one search space",
    source: [80, 81],
    diagram: {
      kind: "search-space",
      title: "Toggle ANN hyperparameters and estimate the search surface",
      description:
        "The search map names the eight source hyperparameters and separates discrete choices from continuous/log-scaled ranges. Trial budgets constrain the experiment cost.",
    },
    objective:
      "List and bound the eight ANN search dimensions and stage architecture tuning before widening the study.",
    intuition:
      "Tuning eight dials at once makes it hard to see which change matters and can multiply compute cost. Begin with a focused pass, then widen with a clear budget.",
    equation:
      "θ = {layers, units, epochs, optimizer, lr, batch size, dropout, weight decay}",
    numerical:
      "If a toy space has 3 layer counts × 5 widths × 3 optimizers × 4 batch sizes, those four dimensions alone make 180 combinations before learning rate, epochs, dropout and decay.",
    shapes: [
      {
        name: "Architecture",
        shape: "layers, width",
        meaning: "Determines model depth and hidden feature size.",
      },
      {
        name: "Optimization",
        shape: "optimizer, lr, epochs",
        meaning: "Controls update rule and training budget.",
      },
      {
        name: "Regularization/data",
        shape: "dropout, weight decay, batch size",
        meaning: "Changes capacity constraints and batch behavior.",
      },
    ],
    sections: [
      section("eight", "Name all eight dimensions", [
        "The source tunes hidden-layer count, neurons per layer, epochs, optimizer, learning rate, batch size, dropout rate and weight decay. It first tunes architecture, then expands to all eight parameters.",
        "Choose bounds from the problem and available resources. Suggest learning rate and weight decay on log scales when orders of magnitude matter. Batch size and optimizer are categorical; layer count and width can be integer choices. Epoch count is a resource budget and interacts with early stopping.",
      ]),
      section("budget", "Constrain model size and trial cost", [
        "A large layer count multiplied by wide layers may exhaust memory or make a trial too slow. Use bounded widths, modest synthetic or stratified subsets for API exploration, and a clear maximum trial/epoch budget.",
        "Optuna may optimize epochs as a parameter, but comparing trials with different compute budgets raises an interpretation question. Report the budget and consider pruning or fixed-resource comparisons. The source's two-round strategy is a practical way to reduce an initially broad search.",
      ]),
    ],
    snippet: "optuna-synthetic",
    template: "optuna-ann-safe",
    additional: ["study-experiments-log"],
    pitfalls: [
      "Search-space size and compute grow quickly with dimensions.",
      "Log-scale parameters should be declared with log=True where appropriate.",
      "Epoch count changes compute budget as well as fit.",
      "Do not tune all dimensions without a stated resource cap.",
    ],
    production:
      "Version search bounds and estimate worst-case memory/time before launching expensive trials. Log pruned and failed trials too.",
    revision: [
      "The source enumerates eight ANN hyperparameters.",
      "Architecture-first search can simplify an initial experiment.",
      "Trial count × training cost determines search expense.",
      "Record bounds and stopping/resource rules.",
    ],
    questions: [
      [
        "Name the eight source hyperparameters.",
        "Hidden layers, units per layer, epochs, optimizer, learning rate, batch size, dropout rate and weight decay.",
      ],
      [
        "Which search values commonly use logarithmic ranges?",
        "Learning rate and often weight decay.",
      ],
      [
        "Why is epoch count special in a study?",
        "It changes both model fit and per-trial compute budget.",
      ],
      [
        "What should happen before a large search?",
        "Bound architecture and estimate compute/memory costs.",
      ],
    ],
  }),
  makeLesson({
    id: "dynamic-ann",
    title: "Construct ANN layers from each trial configuration",
    subtitle: "A dynamic model must update feature width at every layer",
    source: [82, 85],
    diagram: {
      kind: "search-space",
      title: "Build a trial-specific hidden stack",
      description:
        "The selected depth and width determine repeated Linear → BatchNorm → ReLU → Dropout blocks. The final classifier always maps the last width to class logits.",
    },
    objective:
      "Build a dynamic Sequential ANN from suggested depth and width while checking layer shapes and train/eval behavior.",
    intuition:
      "Each hidden layer becomes the next layer's input. Updating input_dim is like changing the adapter size to fit the piece just produced.",
    equation:
      "d₀=784; for each hidden block: Linear(dᵢ,H), update dᵢ₊₁=H; final Linear(d_last,10)",
    numerical:
      "For three hidden layers of width 64, layer widths are 784→64, 64→64, 64→64, then 64→10. A batch of 16 ends with logits [16,10].",
    shapes: [
      { name: "Input", shape: "[B,784]", meaning: "Flattened Fashion-MNIST pixels." },
      {
        name: "Hidden stack",
        shape: "[B,H] repeated L times",
        meaning: "Trial-specific blocks with normalization, activation and dropout.",
      },
      { name: "Classifier", shape: "[B,10]", meaning: "Class logits." },
    ],
    sections: [
      section("build", "Append registered modules in a loop", [
        "Create a Python list of modules. For each hidden layer append Linear(input_dim,width), BatchNorm1d(width), ReLU and Dropout(p); then update input_dim=width. Construct nn.Sequential(*layers) and append the final classifier Linear(input_dim,num_classes).",
        "Keep the final layer free of BatchNorm, ReLU and Dropout for a standard CrossEntropyLoss classifier. If the model includes BatchNorm, ensure training batches can support its statistics.",
      ]),
      section("objective", "Keep one lifecycle inside each isolated trial", [
        "The source places data loaders, model construction, loss, optimizer, training loop and evaluation inside objective(trial). Re-create model and optimizer for each trial so learned weights do not leak across trials. Use validation rather than test data to return the objective.",
        "The runnable objective example follows this corrected split protocol and performs one final best-configuration refit before test reporting. It uses synthetic data and CPU settings so the code can be executed without dataset downloads.",
      ]),
    ],
    snippet: "optuna-ann-safe",
    template: "regularized-ann",
    additional: ["optuna-synthetic"],
    pitfalls: [
      "Update input_dim after adding each hidden layer.",
      "Reinitialize model weights and optimizer state for every trial.",
      "Return validation metric, not test metric, from objective.",
      "Do not add an output activation before CrossEntropyLoss.",
    ],
    production:
      "Log the sampled architecture as a serializable config and reconstruct it from that config when loading or refitting the selected model.",
    revision: [
      "Build repeated hidden blocks dynamically.",
      "The classifier maps the last hidden width to class count.",
      "Each trial starts with fresh weights and optimizer state.",
      "Trial selection uses validation data.",
    ],
    questions: [
      [
        "Why update input_dim inside the loop?",
        "The next Linear layer must accept the previous hidden layer's output width.",
      ],
      ["What is the output shape for B=16 and 10 classes?", "[16,10] raw logits."],
      [
        "Should an Optuna trial reuse the previous trial's weights?",
        "No. Trials should represent independently initialized configurations unless warm-starting is explicitly part of the experiment.",
      ],
      [
        "Which split supplies the trial value?",
        "Validation, with test reserved for a final report.",
      ],
    ],
  }),
  makeLesson({
    id: "ann-objective",
    title: "Tune, refit the best ANN and evaluate test once",
    subtitle: "A leakage-safe training objective with reproducible choices",
    source: [81, 88],
    diagram: {
      kind: "trial-comparison",
      title: "Compare validation trials, refit best, then open test once",
      description:
        "The objective never receives test data. After selecting and refitting the best validation configuration, evaluate the held-out test split once for a final report.",
    },
    objective:
      "Implement the ANN trial lifecycle without test leakage and distinguish best observed parameters from guaranteed best settings.",
    intuition:
      "The test set is the sealed final exam. If every trial checks its answer key, the final score no longer measures unseen performance.",
    equation:
      "trial → train(train) → score(validation) → select → refit → evaluate(test once)",
    numerical:
      "Two trials on 100 train / 40 validation records return validation loss. Refit the better setup on allowed development data and report one result over 40 untouched test records.",
    shapes: [
      {
        name: "Trial training",
        shape: "[Ntrain,784]",
        meaning: "Only training rows update weights.",
      },
      {
        name: "Trial metric",
        shape: "validation scalar",
        meaning: "Used to compare configurations.",
      },
      {
        name: "Final evaluation",
        shape: "test predictions [Ntest,C]",
        meaning: "One final score after selection/refit.",
      },
    ],
    sections: [
      section("lifecycle", "Initialize and train from scratch for each trial", [
        "A trial suggests architecture and optimization choices, makes a fresh model and optimizer, trains only on training data, then computes a validation metric in eval/inference mode. Batch size may be tuned by building a new train loader for that trial.",
        "A minimal testable study can tune fewer dimensions first. Fix seeds at model initialization and loader shuffling; record the sampler seed too. If using pruners, report intermediate validation values at consistent steps.",
      ]),
      section("final", "Refit and leave the test set outside the objective", [
        "After the study ends, take its best observed parameters and run a clearly defined final fit using train plus any permitted validation data, or refit on the development split according to the protocol. Then evaluate the final test set once. Do not return test accuracy from objective(trial).",
        "The source's ANN example evaluates on a split named test_loader inside every trial and reports study results from those scores. That is test leakage for model selection; the corrected flow returns validation loss and holds test out.",
      ]),
      section("tracking", "Report the run rather than a headline alone", [
        "Save trial parameters, states and values, the best observed configuration, seed, split definitions, epoch/resource budget and software versions. Optuna supports storage and visualizations; plotting extras are optional. MLflow integration is optional and is not required to run these examples.",
        "Source results near 89–90% and the observed preference for 2–3 layers, wider layers, 40–50 epochs, SGD and lower dropout are findings from its reported setup, not universal recommendations.",
      ]),
    ],
    snippet: "optuna-ann-safe",
    template: "optuna-ann-safe",
    additional: ["optuna-synthetic", "optuna-plotting", "study-experiments-log"],
    pitfalls: [
      "Never put test data in the objective's closure, loader or metrics.",
      "Rebuild the model and optimizer each trial.",
      "Report that the study gives the best observed result under a budget.",
      "Do not copy source tuning preferences to a different dataset as rules.",
    ],
    production:
      "Keep immutable data split identifiers, persisted study metadata and selected checkpoint provenance. Make final test evaluation a separate reviewed step.",
    revision: [
      "Each trial trains on train and returns validation only.",
      "Fresh models isolate trial weight initialization.",
      "Refit the selected configuration under a stated protocol.",
      "Evaluate held-out test once after choices are done.",
    ],
    questions: [
      [
        "Why is test accuracy inside every trial a problem?",
        "It lets test results influence configuration selection, so the final estimate is no longer independent.",
      ],
      [
        "What should objective return?",
        "A validation metric with direction declared on the study.",
      ],
      [
        "What must reset between trials?",
        "Model initialization and optimizer state, plus any trial-specific loader configuration.",
      ],
      [
        "Does best_params prove global optimality?",
        "No. It is the best observed configuration among the sampled trials and budget.",
      ],
    ],
  }),
];

export const phase06Snippets = snippets;
