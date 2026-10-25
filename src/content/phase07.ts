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
  id, title, purpose, dependencies, shapes, code, expected,
  lines: code.split("\n").map((line, i) => {
    const text = line.trim();
    const note = text.startsWith("import ") ? "Import a dependency used below."
      : text.startsWith("class ") ? "Define a reusable PyTorch module."
      : text.startsWith("def ") ? "Make the calculation explicit and reusable."
      : text.startsWith("assert ") ? "Verify a numerical or shape invariant."
      : text.startsWith("print(") ? "Display the checked result."
      : text.startsWith("#") ? text.replace(/^#+\s*/, "")
      : `Trace this step through ${shapes}.`;
    return `Line ${i + 1} · ${note}`;
  }),
});

const snippets: Snippet[] = [
  makeSnippet("cnn-locality", "Compare a flattened ANN with a local image patch",
    "Show that flattening changes representation and that a dense layer can connect every pixel to every unit.",
    "image [1,1,4,4] → flat [1,16] → local patch [1,1,2,2]",
    `import torch

image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
flat = image.flatten(start_dim=1)
patch = image[:, :, 1:3, 1:3]
ann_parameters = (4 * 4 + 1) * 8
shared_filter_parameters = 3 * 3 + 1
assert flat.shape == (1, 16) and patch.shape == (1, 1, 2, 2)
assert ann_parameters == 136 and shared_filter_parameters == 10
print("flat:", flat.shape, "patch:", patch.shape)
print("dense weights:", ann_parameters, "one shared filter:", shared_filter_parameters)`,
    "flat: torch.Size([1, 16]) patch: torch.Size([1, 1, 2, 2]); dense weights 136; one 3×3 filter with bias 10. The flatten operation itself preserves values; it removes the explicit grid axes."),
  makeSnippet("image-tensor-shapes", "Convert grayscale and RGB batches to NCHW",
    "Assert the channel-first input contract used by Conv2d.",
    "grayscale [N,1,H,W], RGB [N,3,H,W]",
    `import torch

gray = torch.zeros(8, 28, 28, dtype=torch.float32).unsqueeze(1)
rgb = torch.zeros(8, 32, 32, 3, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
assert gray.shape == (8, 1, 28, 28)
assert rgb.shape == (8, 3, 32, 32)
assert gray.dtype == rgb.dtype == torch.float32
print("NCHW:", tuple(gray.shape), tuple(rgb.shape))`,
    "NCHW: (8, 1, 28, 28) (8, 3, 32, 32)"),
  makeSnippet("manual-cross-correlation", "Compute a valid cross-correlation by hand",
    "Implement the sliding multiply-and-sum operation and compare every output against PyTorch.",
    "input [1,1,4,4], kernel [1,1,3,3] → output [1,1,2,2]",
    `import torch
import torch.nn.functional as F

x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
weight = torch.tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.float32).reshape(1, 1, 3, 3)
manual = torch.empty(2, 2, dtype=torch.float32)
for row in range(2):
    for col in range(2):
        patch = x[0, 0, row:row + 3, col:col + 3]
        manual[row, col] = (patch * weight[0, 0]).sum()
library = F.conv2d(x, weight)
assert torch.equal(manual, library[0, 0])
assert torch.equal(manual, torch.full((2, 2), -6.0))
print("manual output:", manual.tolist())`,
    "tensor([[-6., -6.], [-6., -6.]]) · PyTorch Conv2d implements cross-correlation (no kernel flip)."),
  makeSnippet("conv2d-basics", "Verify Conv2d weight sharing and output channels",
    "Inspect the weight layout and prove that every filter produces one output channel.",
    "x [2,1,5,5], Conv2d(1,2,3) → y [2,2,3,3]",
    `import torch
from torch import nn

layer = nn.Conv2d(1, 2, kernel_size=3, bias=True)
x = torch.randn(2, 1, 5, 5, dtype=torch.float32)
y = layer(x)
assert layer.weight.shape == (2, 1, 3, 3)
assert y.shape == (2, 2, 3, 3)
assert sum(p.numel() for p in layer.parameters()) == 20
print("weights:", tuple(layer.weight.shape), "output:", tuple(y.shape), "parameters:", 20)`,
    "weights (2,1,3,3), output (2,2,3,3), trainable parameters 20."),
  makeSnippet("output-size-formula", "Check convolution output sizes against PyTorch",
    "Use the dilation-aware floor formula and compare representative settings to Conv2d.",
    "height/width n,k,p,s,d → floor((n+2p-d(k-1)-1)/s + 1)",
    `import torch
from torch import nn

def output_size(n, k, p, s, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

for n, k, p, s, d in [(28, 3, 1, 1, 1), (28, 3, 0, 1, 1), (28, 3, 1, 2, 1), (15, 3, 2, 1, 2)]:
    layer = nn.Conv2d(1, 2, k, stride=s, padding=p, dilation=d)
    y = layer(torch.zeros(1, 1, n, n, dtype=torch.float32))
    assert y.shape[-2:] == (output_size(n, k, p, s, d),) * 2
assert output_size(28, 3, 1, 1) == 28
assert output_size(28, 3, 1, 2) == 14
print("same:", output_size(28, 3, 1, 1), "stride 2:", output_size(28, 3, 1, 2))`,
    "Same-padded stride 1 gives 28; stride 2 gives 14. Dilation expands the effective kernel to d(k−1)+1."),
  makeSnippet("multichannel-conv", "Trace RGB channels through multiple filters",
    "Check Conv2d's [out_channels,in_channels,kH,kW] parameter tensor and parameter formula.",
    "x [4,3,32,32] → Conv2d(3,16,3,p=1) → [4,16,32,32]",
    `import torch
from torch import nn

layer = nn.Conv2d(3, 16, kernel_size=3, padding=1)
x = torch.randn(4, 3, 32, 32, dtype=torch.float32)
y = layer(x)
expected_parameters = ((3 * 3 * 3) + 1) * 16
assert tuple(layer.weight.shape) == (16, 3, 3, 3)
assert y.shape == (4, 16, 32, 32)
assert sum(p.numel() for p in layer.parameters()) == expected_parameters == 448
print("output:", tuple(y.shape), "parameters:", expected_parameters)`,
    "Output (4,16,32,32), 448 convolution parameters. Spatial dimensions do not affect this count."),
  makeSnippet("cnn-activations", "Check ReLU and BatchNorm2d channel parameters",
    "Verify elementwise ReLU behavior and one affine scale and shift per channel.",
    "x [N,C,H,W] → ReLU → BatchNorm2d(C), trainable affine parameters 2C",
    `import torch
from torch import nn

x = torch.tensor([[[[-2.0, 3.0]], [[4.0, -5.0]]]], dtype=torch.float32)
relu = nn.ReLU()(x)
bn = nn.BatchNorm2d(2)
bn.train()
y = bn(relu)
assert torch.equal(relu, torch.tensor([[[[0.0, 3.0]], [[4.0, 0.0]]]]))
assert y.shape == x.shape
assert sum(p.numel() for p in bn.parameters()) == 4
assert bn.running_mean.shape == (2,)
print("ReLU:", relu.flatten().tolist(), "BatchNorm affine parameters:", 4)`,
    "ReLU zeros negatives; BatchNorm2d(2) has 4 affine parameters (γ and β per channel), plus non-trainable running statistics."),
  makeSnippet("pooling-ops", "Compare max, average and adaptive global pooling",
    "Calculate fixed 2×2 pooling outputs and collapse each channel to one value.",
    "map [1,1,4,4] → pool [1,1,2,2] → adaptive average [1,1,1,1]",
    `import torch
from torch import nn

x = torch.tensor([[[[1, 3, 2, 0], [9, 2, 6, 8], [4, 1, 3, 2], [0, 5, 7, 1]]]], dtype=torch.float32)
maximum = nn.MaxPool2d(2, 2)(x)
average = nn.AvgPool2d(2, 2)(x)
global_average = nn.AdaptiveAvgPool2d((1, 1))(x)
assert maximum.flatten().tolist() == [9, 8, 5, 7]
assert torch.allclose(average.flatten(), torch.tensor([3.75, 4.0, 2.5, 3.25]))
assert global_average.shape == (1, 1, 1, 1)
assert torch.isclose(global_average.squeeze(), torch.tensor(3.375))
print("max:", maximum.flatten().tolist(), "average:", average.flatten().tolist(), "global:", global_average.item())`,
    "Max [9,8,5,7]; average [3.75,4,2.5,3.25]; global average 3.375. Pooling has no learned parameters."),
  makeSnippet("receptive-field-calc", "Calculate receptive-field growth",
    "Apply the receptive-field and jump recurrences across two 3×3 convolutions and a 2×2 pool per block.",
    "r_out = r_in + (k−1)j_in; j_out = j_in s",
    `import torch

def add_layer(receptive_field, jump, kernel, stride):
    return receptive_field + (kernel - 1) * jump, jump * stride

r, j = 1, 1
trace = []
for _ in range(2):
    r, j = add_layer(r, j, 3, 1)
    r, j = add_layer(r, j, 3, 1)
    r, j = add_layer(r, j, 2, 2)
    trace.append((r, j))
assert trace == [(6, 2), (16, 4)]
print("after block 1:", trace[0], "after block 2:", trace[1])`,
    "Starting at receptive field 1 and jump 1: after block one (6,2); after block two (16,4). Padding affects border alignment."),
  makeSnippet("cnn-model", "Build the two-block Fashion-MNIST CNN",
    "Use registered modules, source architecture widths, correct classifier size, and logits for CrossEntropyLoss.",
    "input [B,1,28,28] → features [B,64,7,7] → logits [B,10]",
    `import torch
from torch import nn

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

torch.manual_seed(7)
model = FashionCNN()
model.eval()
with torch.inference_mode():
    logits = model(torch.zeros(2, 1, 28, 28, dtype=torch.float32))
assert logits.shape == (2, 10)
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 429450
print("logits:", tuple(logits.shape), "trainable:", 429450)`,
    "Logits shape (2,10); trainable parameters 429,450. CrossEntropyLoss consumes logits and long class IDs."),
  makeSnippet("cnn-transform", "Reshape flat CSV pixels into normalized images",
    "Provide a safe tensor-only alternative for local Fashion-MNIST CSV rows.",
    "pixels [B,784] in [0,255] → images [B,1,28,28] float32 in [0,1]",
    `import torch

pixels = torch.arange(2 * 784, dtype=torch.float32).reshape(2, 784).remainder(256)
images = (pixels / 255.0).reshape(-1, 1, 28, 28)
labels = torch.tensor([3, 7], dtype=torch.long)
assert images.shape == (2, 1, 28, 28)
assert images.dtype == torch.float32 and labels.dtype == torch.long
assert images.min() >= 0 and images.max() <= 1
print("images:", tuple(images.shape), images.dtype, "labels:", labels.tolist())`,
    "images (2,1,28,28) float32 in [0,1], labels [3,7] as int64/long."),
  makeSnippet("cnn-shape-trace", "Trace every CNN stage and count parameters",
    "Use a forward hook to inspect output shapes while calling the model normally.",
    "[B,1,28,28] → [B,32,14,14] → [B,64,7,7] → [B,10]",
    `import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
)
shapes = {}
handles = [layer.register_forward_hook(lambda module, inputs, output, i=i: shapes.update({i: tuple(output.shape)}))
          for i, layer in enumerate(model) if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Flatten, nn.Linear))]
model.eval()
with torch.inference_mode():
    logits = model(torch.zeros(2, 1, 28, 28, dtype=torch.float32))
for handle in handles:
    handle.remove()
assert logits.shape == (2, 10)
assert shapes[0] == (2, 32, 28, 28) and shapes[7] == (2, 64, 7, 7)
assert sum(p.numel() for p in model.parameters()) == 429450
print("traced stages:", shapes, "total:", sum(p.numel() for p in model.parameters()))`,
    "Shape trace reaches (2,3136), (2,128), (2,64), (2,10); total 429,450. Hooks inspect model(x) without changing forward."),
  makeSnippet("cnn-smoke-train", "Run one synthetic CNN training epoch",
    "Exercise train mode, CrossEntropyLoss, optimizer updates, and eval mode without downloads.",
    "images [8,1,28,28], labels [8] long → logits [8,10]",
    `import torch
from torch import nn

torch.manual_seed(19)
torch.set_num_threads(1)
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 7 * 7, 10),
)
x = torch.rand(8, 1, 28, 28, dtype=torch.float32)
y = torch.arange(8, dtype=torch.long) % 10
optimizer, loss_fn = torch.optim.SGD(model.parameters(), lr=0.01), nn.CrossEntropyLoss()
model.train()
for start in range(0, 8, 4):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x[start:start + 4])
    loss = loss_fn(logits, y[start:start + 4])
    loss.backward()
    optimizer.step()
model.eval()
with torch.inference_mode():
    scores = model(x)
assert scores.shape == (8, 10) and torch.isfinite(scores).all()
print("one synthetic epoch complete:", tuple(scores.shape))`,
    "one synthetic epoch complete: (8,10). This small model verifies the training path, not Fashion-MNIST accuracy."),
  makeSnippet("fashionmnist-cnn-optional", "Optional Fashion-MNIST training",
    "Run a reproducible torchvision baseline only after explicitly enabling the download flag.",
    "Dataset images [B,1,28,28], labels [B] long → logits [B,10]",
    `import torch
from torch import nn
from torch.utils.data import DataLoader

DOWNLOAD_DATA = False  # Set True only when you intend to fetch Fashion-MNIST.
if not DOWNLOAD_DATA:
    raise SystemExit("Set DOWNLOAD_DATA=True to download the optional dataset.")
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
).to(device)
optimizer, loss_fn = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4), nn.CrossEntropyLoss()
for epoch in range(1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
model.eval()
correct = total = 0
with torch.inference_mode():
    for images, labels in test_loader:
        logits = model(images.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += labels.numel()
assert total == len(test_data)
print(f"one-epoch test accuracy: {correct / total:.3f} ({correct}/{total})")`,
    "Optional one-epoch run prints observed test accuracy. The dataset download is guarded by an explicit flag; use validation data for model selection and reserve test for final reporting.",
    "Python 3.10+ · PyTorch · torchvision · CPU or CUDA · download only when explicitly enabled"),
  makeSnippet("cnn-shape-debug", "Diagnose a common CNN shape mismatch",
    "Turn a Linear input-size error into a measured feature shape and derived in_features.",
    "feature [B,64,7,7] → flatten [B,3136] → Linear(3136,10)",
    `import torch
from torch import nn

features = torch.zeros(4, 64, 7, 7, dtype=torch.float32)
flat = nn.Flatten()(features)
head = nn.Linear(flat.shape[1], 10)
logits = head(flat)
assert flat.shape == (4, 3136) and logits.shape == (4, 10)
assert head.in_features == 64 * 7 * 7
print("measured in_features:", flat.shape[1], "logits:", tuple(logits.shape))`,
    "measured in_features: 3136 logits: (4,10). Derive the dense width from the actual feature output instead of guessing."),
];
export const phase07Snippets = snippets;

const byId = Object.fromEntries(snippets.map((snippet) => [snippet.id, snippet]));
const makeLesson = (
  id: string, title: string, subtitle: string, pages: [number, number], diagram: Lesson["diagram"],
  objectives: string[], intuition: string, equation: string, numerical: string,
  shapes: Lesson["shapes"], sections: Lesson["sections"], main: string, template: string,
  extras: string[], pitfalls: string[], production: string, revision: string[],
  questions: [string, string][],
): Lesson => ({
  id, moduleId: "convolution", title, subtitle, minutes: 9,
  source: { start: pages[0], end: pages[1], kind: "source-derived",
    note: "Core concept follows the source; any engineering extension is labeled supplemental." },
  objectives, intuition, equation, numerical, shapes, diagram, sections,
  snippet: byId[main], template: byId[template], additionalSnippets: extras.map((key) => byId[key]),
  pitfalls, production, revision, questions: questions.map(([question, answer]) => ({ question, answer })),
});

export const phase07Lessons: Lesson[] = [
  makeLesson("cnn-motivation", "Keep the neighborhood in view",
    "Why image models use local connectivity and shared filters.", [89, 101],
    { kind: "cnn-locality", title: "A grid has neighbors; a dense layer has positions",
      description: "A pixel grid and flattened vector contain the same values, but the CNN keeps the two spatial axes explicit and reuses a local filter across positions." },
    ["Explain what flattening changes and what it preserves.", "Compare dense connectivity with a shared local filter.", "Describe translation equivariance without claiming perfect invariance."],
    "Flattening is like writing a map's addresses in one long list: the coordinates can be reconstructed, but adjacency is no longer built into the operation. A convolution uses a small stencil repeatedly, like applying one reusable stamp across a tiled floor.",
    "Dense layer parameters = (H·W·C + 1)·units; one convolution filter parameters = (kH·kW·Cin + 1).",
    "For a 4×4 grayscale image and 8 dense units, the first layer has (16+1)×8=136 parameters. A shared 3×3 grayscale filter has 9 weights and one bias: 10 parameters.",
    [{ name: "Image batch", shape: "[1,1,4,4]", meaning: "Batch, channel, height, width." },
     { name: "Flattened ANN input", shape: "[1,16]", meaning: "Same pixel values, with grid axes collapsed." },
     { name: "Local image patch", shape: "[1,1,2,2]", meaning: "A spatial neighborhood stays explicit." }],
    [{ id: "locality", title: "Flattening changes the inductive bias",
       paragraphs: ["Flattening is a reshape: it does not delete pixel values. A dense layer can learn positional relationships, but it starts with separate weights for separate positions. A CNN starts with local receptive fields and shares each filter across locations, which is a useful image prior and can greatly reduce parameters.",
       "The same filter at different positions gives translation equivariance away from boundaries: shifting an input tends to shift its feature map. Stride, padding, pooling, and finite boundaries break exact behavior. A CNN is not automatically invariant to arbitrary shifts or rotations."] },
     { id: "comparison", title: "The Fashion-MNIST motivation",
       paragraphs: ["The PDF reports earlier ANN results from roughly 83% to 90.2% test accuracy and a basic CNN at 92.62%, with 99.99% training accuracy. These are reported results for that run, not a guaranteed benchmark; the large train/test gap is evidence of remaining overfit.",
       "A convolution does not make every model smaller in every setting. Compare the actual parameter and compute budget, and preserve the dataset's class and split protocol when evaluating."] }],
    "cnn-locality", "cnn-locality", [], ["Flattening is not information deletion; it removes explicit spatial axes.", "Do not describe CNNs as perfectly translation or rotation invariant.", "Dense layers can model images, but do not share local weights by default.", "Reported PDF accuracy is a run-specific result."],
    "Use augmentation and validation-based selection when appropriate; test only after model choices are complete.",
    ["Convolution encodes local connectivity and weight sharing.", "Flatten preserves values but changes the model's prior.", "Shared filters support equivariance, not universal invariance.", "Compare held-out metrics and the train/validation gap."],
    [["Does flattening destroy pixel values?", "No. It changes their layout and removes explicit H/W axes; values and ordering remain."],
     ["Why does a shared filter use fewer parameters?", "The same kH×kW×Cin weights are reused at every spatial position."],
     ["Does a CNN guarantee translation invariance?", "No. Convolution is approximately equivariant under suitable conditions; pooling may add local robustness, but the whole model is not universally invariant."],
     ["What should you infer from 99.99% train and 92.62% test?", "The source run fits training data much more strongly than held-out data, indicating overfitting."]]),
  makeLesson("image-tensors", "Read an image tensor",
    "Represent grayscale and color images with PyTorch's channel-first convention.", [90, 94],
    { kind: "cnn-locality", title: "From pixels to NCHW",
      description: "The first axis counts examples, the second channels, and the final axes are image height and width." },
    ["Distinguish sample shape CHW from batch shape NCHW.", "Convert RGB channels-last input to PyTorch channel-first layout.", "Normalize byte intensities and preserve label dtype."],
    "Think of a batch as a stack of albums, each image as a stack of channel transparencies, and every channel as a height-by-width grid.",
    "PyTorch Conv2d input: [N,Cin,H,W]; grayscale C=1, RGB C=3.",
    "A batch of 32 grayscale 28×28 images is [32,1,28,28]. A single RGB image 32×32 is [3,32,32]; four such images are [4,3,32,32].",
    [{ name: "One grayscale image", shape: "[1,28,28]", meaning: "Channel, height, width." },
     { name: "Batch", shape: "[32,1,28,28]", meaning: "N, C, H, W for Conv2d." },
     { name: "RGB conversion", shape: "[4,32,32,3] → [4,3,32,32]", meaning: "Permute HWC into NCHW." }],
    [{ id: "axes", title: "The axes carry meaning",
       paragraphs: ["For an individual image, use [C,H,W]. A DataLoader batches those samples into [N,C,H,W]. Fashion-MNIST is grayscale with one channel; RGB data has three channels. Internal CNN feature maps use channels for learned features, not necessarily color.",
       "The PDF reshapes flat CSV rows from 784 pixels into [N,1,28,28] and scales 0–255 to 0–1. For torchvision, ToTensor performs this layout conversion and scaling for common image inputs; inspect actual dtype and range when adapting transforms."] },
     { id: "dataset", title: "Keep examples and labels paired",
       paragraphs: ["The source constructs a custom Dataset and DataLoader, shuffling training examples and retaining ordered evaluation. Use float32 images and long class indices for CrossEntropyLoss. The final batch can be smaller than the requested batch size.",
       "The code export contains clipped CSV and loader lines. The accompanying example reconstructs only the verified shape/scaling contract; no source URL or incomplete code is copied."] }],
    "image-tensor-shapes", "image-tensor-shapes", ["cnn-transform"], ["Conv2d expects NCHW by default.", "Do not confuse a single image [C,H,W] with a batch [N,C,H,W].", "Use float32 normalized pixels; CrossEntropy class targets are long.", "Check if a source is HWC before permuting."],
    "Normalize from training data consistently and apply the same transform to validation/test. For image augmentation, random transforms belong only in the training pipeline.",
    ["Single image: CHW; batch: NCHW.", "Grayscale has one channel; RGB has three.", "Scale byte pixels to a documented range.", "Labels for ten-way cross entropy have shape [N] and dtype long."],
    [["What does N mean in NCHW?", "The number of images in the batch."],
     ["What is a grayscale Fashion-MNIST batch shape?", "[N,1,28,28]."],
     ["Why permute RGB [N,H,W,3]?", "Conv2d expects channels before height and width: [N,3,H,W]."],
     ["What target dtype does CrossEntropyLoss expect?", "Integer class indices, normally torch.long, shaped [N]."]]),
  makeLesson("convolution-math", "Slide, multiply, and sum",
    "Understand the Conv2d operation as channel-wise cross-correlation plus bias.", [98, 105],
    { kind: "conv-grid", title: "A kernel reads one patch at a time",
      description: "Move the highlighted 3×3 window over the 5×5 grid. Each output cell is the sum of nine elementwise products and a bias." },
    ["Calculate a single valid convolution output numerically.", "Explain the learned bias and output feature map.", "Distinguish mathematical convolution from PyTorch cross-correlation."],
    "A small stencil reads a local patch, multiplies each position by a learned coefficient, sums the products, adds bias, and writes one output pixel.",
    "Y[n,o,i,j] = b[o] + Σc,u,v X[n,c,i·s−p+u·d,j·s−p+v·d] W[o,c,u,v].",
    "Patch [[1,2,0],[0,1,2],[2,0,1]] with vertical-edge kernel [[1,0,−1],[1,0,−1],[1,0,−1]] gives 1+0+0+0+0−2+2+0−1=0, before bias.",
    [{ name: "Input", shape: "[1,1,4,4]", meaning: "One grayscale image." },
     { name: "Kernel", shape: "[1,1,3,3]", meaning: "One output channel, one input channel, 3×3 weights." },
     { name: "Valid output", shape: "[1,1,2,2]", meaning: "One sum at each legal filter location." }],
    [{ id: "operation", title: "What one output value means",
       paragraphs: ["At each location, the filter and image patch have matching shapes. Multiply corresponding entries, sum them, and add the output channel's bias. Repeating across positions creates one feature map. Multiple output filters create multiple maps.",
       "Strictly, convolution flips a kernel before sliding; PyTorch Conv2d and F.conv2d use cross-correlation without flipping. Deep-learning literature commonly calls this convolution. Since filter weights are learned, this convention is usually handled by training, but it matters when checking a hand-computed example."] },
     { id: "implementation", title: "Check arithmetic against F.conv2d",
       paragraphs: ["The source's worked matrix arithmetic conflicts with its illustrated feature map on pp.103–104. The sample here uses a separate simple 4×4 input whose four valid outputs are all −6; it computes them in loops and asserts exact agreement with F.conv2d.",
       "Inputs and weights are float32. For a bias-free hand calculation, omit the bias or set it to zero; changing bias adds the same scalar to every output location for that filter."] }],
    "manual-cross-correlation", "conv2d-basics", ["manual-cross-correlation"], ["PyTorch Conv2d applies cross-correlation without flipping the kernel.", "Filter depth must match input channels.", "Bias is one scalar per output channel.", "Check both patch coordinates and kernel orientation in hand arithmetic."],
    "Convolution uses shared parameters but still performs work at each output location. Estimate activation memory and compute as image resolution, channel counts, and batch size grow.",
    ["Multiply corresponding entries, sum, then add bias.", "Weights are reused across spatial locations.", "PyTorch calls cross-correlation Conv2d.", "Verify worked numbers against a library implementation."],
    [["What operation does PyTorch call Conv2d?", "Cross-correlation: it does not flip the kernel."],
     ["Where does the bias apply?", "One learned bias per output filter, added to every spatial location in that output map."],
     ["What are the two output spatial dimensions in the code?", "2×2: a 3×3 kernel fits in a 4×4 input at two positions per axis."],
     ["Why test a hand calculation against F.conv2d?", "It catches orientation, indexing, padding, and arithmetic mistakes."]]),
  makeLesson("convolution-shapes", "Predict the feature-map size",
    "Use padding, stride, and dilation to derive Conv2d output dimensions.", [105, 108],
    { kind: "conv-shapes", title: "Padding and stride change the scan",
      description: "Choose padding and stride to see the formula update the height and width; the 28-pixel examples remain tied to exact integer output sizes." },
    ["Explain valid and same padding for stride one.", "Calculate output height and width with the floor formula.", "Use dilation as an explicitly labeled extension."],
    "Padding adds a border for the filter to read; stride is the number of input pixels between neighboring filter positions. Dilation spaces filter taps apart.",
    "out = floor((n + 2p − d(k−1) − 1)/s + 1); for d=1 this becomes floor((n+2p−k)/s +1).",
    "With n=28,k=3,p=1,s=1,d=1: floor((28+2−3)/1+1)=28. With stride 2 and same p: floor(27/2+1)=14.",
    [{ name: "Input spatial shape", shape: "[28,28]", meaning: "Height and width are calculated independently." },
     { name: "3×3 same, stride 1", shape: "[28,28]", meaning: "Padding one preserves size for this odd kernel." },
     { name: "3×3, padding 1, stride 2", shape: "[14,14]", meaning: "The filter advances two pixels per output." }],
    [{ id: "formula", title: "Count the legal filter positions",
       paragraphs: ["The effective kernel width is d(k−1)+1. The padded extent is n+2p. Subtract the effective kernel, divide by stride, floor, then add one for the initial location. Apply the same calculation separately to height and width.",
       "For dilation one, padding zero is often called valid; positions beyond the input are not considered. For odd k and stride one, p=(k−1)/2 keeps the spatial size. PyTorch padding='same' supports stride one; numeric padding makes the arithmetic explicit."] },
     { id: "controls", title: "Stride can leave a partial final step",
       paragraphs: ["For input 28, kernel 3, padding 1, stride 2, the last filter begins at coordinate 26 and fits; the floor discards any incomplete extra position. A 2×2 max pool with stride 2 maps 28 to 14 and then 7.",
       "Dilation is supplementary here: it inserts gaps between taps without adding weights. The code checks ordinary and dilated cases against actual Conv2d outputs."] }],
    "output-size-formula", "output-size-formula", [], ["The final floor is essential when stride does not divide the usable extent.", "Height and width can differ; calculate each separately.", "Padding='same' is not arbitrary for all stride choices.", "Dilation changes the effective kernel but not the number of weights."],
    "Use runtime assertions around model boundaries when input sizes vary; document any fixed-size assumption before a Flatten→Linear head.",
    ["Compute each spatial axis with the same formula.", "Valid means no padding; same with odd kernels and stride one preserves size.", "Stride controls output spacing and resolution.", "Dilation expands receptive support without adding taps."],
    [["For n=28,k=3,p=1,s=1,d=1, what is out?", "28."],
     ["What is the output for 28, kernel 3, padding 1, stride 2?", "14."],
     ["What does floor do?", "It keeps only fully fitting filter placements at the chosen stride."],
     ["Does dilation add trainable kernel weights?", "No. It spaces the same taps farther apart."]]),
  makeLesson("convolution-channels", "Stack channels, then stack filters",
    "Trace multi-channel filters, output feature maps, Conv2d parameters, and BatchNorm2d.", [109, 112],
    { kind: "conv-channels", title: "One filter spans every input channel",
      description: "Adjust input channels and output filters. Each output filter has Cin×kH×kW weights and one bias; output channels equal filter count." },
    ["Explain why one filter spans the entire input depth.", "Calculate Conv2d and BatchNorm2d trainable parameters.", "Describe ReLU and BatchNorm2d on NCHW tensors."],
    "A filter is a stack of matching 2D stencils. It reads every input channel and sums across depth to produce one map; a bank of filters produces an output stack.",
    "Conv2d params = (Cin·kH·kW + 1)·Cout; affine BatchNorm2d params = 2·Cout.",
    "For RGB input and 16 3×3 filters: (3×3×3+1)×16 = 448 convolution parameters. BatchNorm2d(16) adds 32 trainable scale/shift values.",
    [{ name: "Input volume", shape: "[4,3,32,32]", meaning: "Four RGB images." },
     { name: "Conv2d weights", shape: "[16,3,3,3]", meaning: "16 filters, each spans all 3 input channels." },
     { name: "Output volume", shape: "[4,16,32,32]", meaning: "One feature map per filter with padding one." }],
    [{ id: "filters", title: "Depth matching and output depth",
       paragraphs: ["Each output filter contains Cin slices of size kH×kW and a bias. It multiplies each slice against the matching input channel, sums every product across channels and spatial positions, then adds bias. Cout filters create Cout output maps.",
       "The weight tensor layout is [Cout,Cin,kH,kW]. Parameter count depends on channel and kernel sizes, not on input height or width. For groups greater than one the formula changes; grouped and depthwise convolution are beyond this source section."] },
     { id: "normalization", title: "ReLU and BatchNorm2d",
       paragraphs: ["ReLU applies max(0,x) elementwise, allowing stacked layers to model nonlinear functions. BatchNorm2d accepts [N,C,H,W], tracks per-channel running statistics over batch and spatial positions during training, and has one learned gamma and beta per channel when affine=True.",
       "The source's block order is Conv→ReLU→BatchNorm2d→Pool; preserve that order when tracing its model. A common alternative is Conv→BatchNorm→ReLU, but it is a separate architecture choice. BatchNorm's running buffers are not trainable parameters."] }],
    "multichannel-conv", "multichannel-conv", ["cnn-activations"], ["Filter input depth must equal in_channels.", "out_channels is the number of filters/output maps.", "BatchNorm2d is configured by channels, not height×width.", "Running mean/variance are buffers, not trainable gamma/beta."],
    "When exporting or loading checkpoints, keep channel order and normalization behavior consistent. Small batches can make BatchNorm estimates noisy; choose normalization with workload and batch size in mind.",
    ["Conv weight layout is [Cout,Cin,kH,kW].", "One filter sums products across all input channels.", "Parameter count is independent of image H and W.", "BatchNorm2d has 2C affine trainable values when affine=True."],
    [["If Cin=3, what is the depth of one 3×3 filter?", "It has three 3×3 slices: one for each input channel."],
     ["What does Cout determine?", "The number of distinct filters and output feature-map channels."],
     ["How many Conv2d parameters for 3 input, 16 output, 3×3, bias enabled?", "448."],
     ["How does BatchNorm2d compute normalization?", "Per channel, using batch and spatial values, with learned scale and shift plus running-stat buffers."]]),
  makeLesson("cnn-activations", "Add nonlinear feature processing",
    "See how ReLU and channel-wise BatchNorm2d change feature maps.", [111, 112],
    { kind: "conv-channels", title: "Channels carry feature statistics",
      description: "A convolution creates feature channels; ReLU gates negative values and BatchNorm2d normalizes each channel using batch and spatial statistics." },
    ["Apply ReLU to positive and negative feature values.", "Trace BatchNorm2d input and output shapes.", "Count affine parameters and distinguish buffers."],
    "ReLU is a one-way gate: positive signals pass, negative signals become zero. BatchNorm2d adjusts each feature channel using learned scale and shift.",
    "ReLU(x)=max(0,x); BatchNorm2d affine parameters = γ[C] + β[C] = 2C.",
    "The feature values [−12,45,−3] become [0,45,0]. For C=32, BatchNorm2d has 64 trainable affine values.",
    [{ name: "Before activation", shape: "[N,C,H,W]", meaning: "Raw convolution maps." },
     { name: "After ReLU and BN", shape: "[N,C,H,W]", meaning: "Same shape; values/statistics change." },
     { name: "BatchNorm2d parameters", shape: "2C", meaning: "Learned gamma and beta, not running buffers." }],
    [{ id: "relu", title: "Nonlinearity makes depth useful",
       paragraphs: ["A sequence of only linear operations can be represented as one linear operation. ReLU makes feature processing nonlinear by zeroing negative activations and keeping positive ones.",
       "ReLU does not change tensor shape or learn parameters. It can produce zero activations; monitor dead units if a model or learning rate is poorly configured."] },
     { id: "batchnorm", title: "Statistics belong to each channel",
       paragraphs: ["For each channel, BatchNorm2d computes mini-batch statistics across N, H, and W during training, normalizes values, then applies learned gamma and beta. In evaluation it uses running estimates accumulated during training.",
       "The source places BatchNorm2d after ReLU in this CNN. Use train() and eval() consistently; evaluation should not update running statistics or apply dropout masks."] }],
    "cnn-activations", "cnn-activations", ["multichannel-conv"], ["ReLU changes values but preserves NCHW dimensions.", "BatchNorm2d expects the channel count C.", "Call eval() before inference to use running statistics.", "BatchNorm buffers are included in state_dict but not trainable counts."],
    "Track train/eval mode in validation and deployment. If batch size is tiny, compare alternatives such as GroupNorm rather than assuming BatchNorm statistics are reliable.",
    ["ReLU is elementwise max(0,x).", "BatchNorm2d computes channel statistics across N,H,W.", "Affine BatchNorm contributes gamma and beta per channel.", "Evaluation uses running statistics."],
    [["What is ReLU(−3)?", "0."],
     ["Does ReLU alter the tensor's shape?", "No, it is elementwise."],
     ["How many trainable affine parameters does BatchNorm2d(32) have?", "64: one gamma and one beta per channel."],
     ["Which mode uses BatchNorm running statistics?", "Evaluation mode, set with model.eval()."]]),
  makeLesson("pooling", "Downsample without changing channels",
    "Compare max, average, and global average pooling.", [112, 114],
    { kind: "pooling", title: "Four values become one",
      description: "Apply a 2×2 pooling window independently to each channel and compare max and mean outputs; global averaging reduces each feature map to 1×1." },
    ["Calculate a 2×2 max-pool output.", "Compare maximum and arithmetic mean.", "Explain global average pooling and its classifier use."],
    "Pooling summarizes a small neighborhood: max keeps the strongest response; average keeps the local mean. Neither operation learns weights.",
    "2×2 stride-2 pooling halves each spatial dimension and keeps the channel count; AdaptiveAvgPool2d(1) maps [C,H,W] to [C,1,1].",
    "For the top-left window [[1,3],[9,2]], max is 9 and mean is (1+3+9+2)/4=3.75.",
    [{ name: "Input feature map", shape: "[1,1,4,4]", meaning: "One channel slice." },
     { name: "2×2 pooling", shape: "[1,1,2,2]", meaning: "Stride 2 creates four summaries." },
     { name: "Global average", shape: "[1,1,1,1]", meaning: "One average per channel." }],
    [{ id: "fixed", title: "Max and average pooling",
       paragraphs: ["MaxPool2d(2,2) divides each channel into non-overlapping 2×2 windows and keeps one largest value. AvgPool2d averages the values. Both preserve channel count and have zero trainable parameters.",
       "Halving height and width leaves one quarter as many spatial values, a 75% reduction. Local max pooling can make nearby shifts less disruptive, but it does not create full shift, rotation, or distortion invariance."] },
     { id: "adaptive", title: "Global average pooling",
       paragraphs: ["AdaptiveAvgPool2d((1,1)) returns one average for each channel regardless of the incoming H and W. Flattening then gives a C-length vector; a final Linear(C,num_classes) can replace a large fixed-size dense head.",
       "The source classifier instead flattens [64,7,7] to 3136 and uses dense layers. GAP is a supplemental alternative with a different capacity and inductive bias."] }],
    "pooling-ops", "pooling-ops", [], ["Pooling acts independently on channels.", "Max pool chooses a value; average pool computes a mean.", "A 2×2 stride-2 output is floor-sized if dimensions are not divisible.", "Small-shift robustness is not universal invariance."],
    "Pooling has no learned parameters, but changes the spatial detail available to later layers. Adaptive pooling can simplify variable-resolution classifier heads.",
    ["Max pool keeps the largest local response.", "Average pool keeps the local mean.", "Channels do not change under spatial pooling.", "Global average pooling produces one scalar per channel."],
    [["What is the top-left max output in the example?", "9."],
     ["What is its average?", "3.75."],
     ["How many trainable weights does MaxPool2d have?", "Zero."],
     ["What shape follows AdaptiveAvgPool2d(1) for [B,64,H,W]?", "[B,64,1,1]."]]),
  makeLesson("receptive-field", "Follow the receptive field",
    "Track how deeper features combine larger regions of the original image.", [114, 116],
    { kind: "receptive-field", title: "Deeper pixels see broader context",
      description: "Step through convolution and pooling layers to see receptive-field size and input-space jump grow together." },
    ["Define receptive field in input-image coordinates.", "Apply receptive-field and jump recurrences.", "Relate depth to hierarchical feature combinations."],
    "A feature pixel is a summary of the input region that could affect it. Each new filter combines neighboring summaries, widening the region it can see.",
    "r_out = r_in + (k−1)j_in; j_out = j_in·s, starting with r=1,j=1.",
    "Two 3×3 convolutions then 2×2 stride-2 pooling give r=6,j=2. A second such block gives r=16,j=4.",
    [{ name: "Input pixel", shape: "r=1, j=1", meaning: "One original pixel at unit spacing." },
     { name: "After block 1", shape: "r=6, j=2", meaning: "Each feature spans six input pixels; adjacent outputs are two apart." },
     { name: "After block 2", shape: "r=16, j=4", meaning: "Context grows while output spacing increases." }],
    [{ id: "growth", title: "Track size and spacing together",
       paragraphs: ["The receptive field r is the width of input context affecting one feature location. The jump j is the distance, measured in input pixels, between neighboring output locations. A kernel of size k and stride s updates r by (k−1)j and updates j by multiplying by s.",
       "For each block here, apply two k=3,s=1 convolutions followed by k=2,s=2 pooling. Starting at r=1,j=1, the first block ends r=6,j=2; the second ends r=16,j=4. This is a theoretical receptive field; learned weights and nonlinearities determine effective influence."] },
     { id: "hierarchy", title: "Feature hierarchy is a useful analogy",
       paragraphs: ["Early filters can respond to local edges or contrast. Later features combine earlier maps into corners, textures, or parts. These labels are intuition, not guarantees about what a trained filter represents.",
       "Padding creates boundary positions whose context partly comes from zeros. Receptive-field recurrences track nominal span and jump; exact border support needs the layer's padding convention."] }],
    "receptive-field-calc", "receptive-field-calc", [], ["Track jump as well as receptive-field width.", "Pooling changes both receptive field and output spacing.", "Nominal receptive field is not a guarantee that learned weights use every pixel.", "Padding affects edge context."],
    "Use receptive-field calculations to reason about context versus resolution. Validate with saliency or ablation tools before making claims about what features the model learned.",
    ["Convolution expands context by (k−1)j.", "Stride multiplies the jump.", "Pooling can sharply enlarge input-space spacing.", "Effective learned context can be smaller than nominal support."],
    [["What is receptive field?", "The region in the original input that can influence one feature location."],
     ["Why track jump?", "It gives the spacing between neighboring output positions in input coordinates."],
     ["What is r,j after one stated block?", "(6,2)."],
     ["Does a 16×16 theoretical receptive field prove all 16×16 pixels matter equally?", "No. It describes possible support; trained weights and nonlinearities set actual influence."]]),
  makeLesson("cnn-classifier", "Build the two-block classifier",
    "Connect convolutional feature extraction to a dense ten-class logits head.", [90, 97],
    { kind: "cnn-flow", title: "Fashion-MNIST from pixels to logits",
      description: "Follow one batch through two Conv-ReLU-BatchNorm-Pool blocks and the fully connected classifier; counts and shapes match the runnable module." },
    ["Implement the source's two convolution blocks and dense classifier.", "Trace all intermediate NCHW and flattened shapes.", "Train with logits-compatible cross entropy and correct modes."],
    "The feature extractor learns reusable local patterns; the classifier combines the resulting feature vector to choose a clothing class.",
    "28→14→7 spatially; 64·7·7=3136 features; logits [B,10]; CrossEntropyLoss(logits, labels).",
    "For batch 32: [32,1,28,28]→[32,32,14,14]→[32,64,7,7]→[32,3136]→[32,128]→[32,64]→[32,10].",
    [{ name: "Input batch", shape: "[32,1,28,28]", meaning: "Grayscale Fashion-MNIST." },
     { name: "Conv/pool block 1", shape: "[32,32,14,14]", meaning: "32 features; pool halves spatial axes." },
     { name: "Conv/pool block 2", shape: "[32,64,7,7]", meaning: "64 features; pool halves again." },
     { name: "Classifier", shape: "[32,3136]→[32,128]→[32,64]→[32,10]", meaning: "Dense feature head and raw class logits." }],
    [{ id: "architecture", title: "Feature extractor then classifier",
       paragraphs: ["The source model uses Conv2d(1,32,3,padding=1), ReLU, BatchNorm2d(32), and MaxPool2d(2); then Conv2d(32,64,3,padding=1), ReLU, BatchNorm2d(64), and another pool. Spatial sizes go 28→14→7.",
       "Flatten converts [B,64,7,7] into [B,3136]. The head is Linear(3136,128), ReLU, Dropout(0.4), Linear(128,64), ReLU, Dropout(0.4), Linear(64,10). The final outputs are raw logits. Do not append softmax before CrossEntropyLoss."] },
     { id: "parameters", title: "Count what the dense layer costs",
       paragraphs: ["The two convolutions have 320 and 18,496 parameters. Their BatchNorm affine parameters add 64 and 128, so the full feature extractor is 19,008. The first Linear has (3136+1)×128=401,536 parameters; the remaining dense layers add 8,906. Total trainable parameters: 429,450.",
       "The PDF's listed feature terms total 19,008 when BatchNorm is included. Its synthetic trace calls a custom forward_with_shape_trace method directly; the sample uses ordinary model(x) plus temporary hooks so the model API remains standard."] },
     { id: "training", title: "Train and evaluate safely",
       paragraphs: ["During training, call model.train(), clear gradients, compute logits, calculate CrossEntropyLoss against long labels, backpropagate, and step the optimizer. During evaluation, call model.eval() and use inference_mode/no_grad. Dropout is active only in train mode and BatchNorm uses running estimates in eval mode.",
       "The source reports a 92.62% test and 99.99% train accuracy. These are source-run figures, not a target. The optional torchvision script is guarded by DOWNLOAD_DATA=False; the synthetic smoke example verifies the training path without fetching data."] }],
    "cnn-model", "cnn-model", ["cnn-shape-trace", "cnn-smoke-train", "fashionmnist-cnn-optional", "cnn-shape-debug"], ["Linear in_features must equal C×H×W after flatten.", "CrossEntropyLoss expects raw [B,10] logits and [B] long labels.", "Use model.train() for updates and model.eval() for measurement.", "Large train/test gaps can signal overfitting.", "Do not call a custom trace method instead of model(x)."],
    "Keep a validation split for tuning and checkpoint selection. Report the held-out test result only after decisions are final. The optional lab runs one epoch; use a controlled training protocol for meaningful comparisons.",
    ["Convolution/pooling shape: 28→14→7.", "Flatten width is 64×7×7=3136.", "Classifier output is ten logits.", "Total parameters: 429,450 including BatchNorm affine values.", "Use train/eval modes and do not select on test."],
    [["Why is the first Linear input width 3136?", "The preceding feature tensor has 64×7×7 values per example."],
     ["What does CrossEntropyLoss receive?", "Raw logits shaped [B,10] and integer class targets shaped [B]."],
     ["How many parameters are in the feature extractor?", "19,008 including both Conv layers and affine BatchNorm parameters."],
     ["What is the purpose of model.eval()?", "It switches dropout off and BatchNorm to running statistics; it does not disable gradients by itself."]]),
];
