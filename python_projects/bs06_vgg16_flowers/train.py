"""VGG16 transfer-learning demo. Default smoke is synthetic, CPU-only, and offline."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def make_classifier(in_features: int, num_classes: int) -> nn.Module:
    if in_features < 1 or num_classes < 2:
        raise ValueError("in_features must be positive and num_classes must be at least two")
    return nn.Sequential(
        nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(128, num_classes),
    )


def freeze_features(model: nn.Module, unfreeze_last_block: bool = False) -> list[nn.Parameter]:
    """Freeze the VGG feature extractor, optionally unfreezing its final conv block."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if unfreeze_last_block:
        # VGG16's last convolutional block begins at feature index 24.
        for parameter in model.features[24:].parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def split_class_indices(targets: list[int], seed: int = 17) -> tuple[list[int], list[int], list[int]]:
    """Stratified 70/15/15 split. Each class needs at least three nonduplicate images."""
    groups: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        groups.setdefault(int(target), []).append(index)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    rng = random.Random(seed)
    for label in sorted(groups):
        indices = groups[label]
        if len(indices) < 3:
            raise ValueError(f"class {label} needs at least three images for train/validation/test")
        rng.shuffle(indices)
        n_train = max(1, int(len(indices) * 0.70))
        n_val = max(1, int(len(indices) * 0.15))
        if n_train + n_val >= len(indices):
            n_train, n_val = len(indices) - 2, 1
        train.extend(indices[:n_train])
        validation.extend(indices[n_train:n_train + n_val])
        test.extend(indices[n_train + n_val:])
    return train, validation, test


def reject_duplicate_images(samples: list[tuple[str, int]]) -> None:
    """Reject exact duplicate files before splitting so copies cannot cross partitions."""
    seen: dict[str, str] = {}
    for filename, _ in samples:
        digest = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"duplicate image bytes found: {seen[digest]} and {filename}")
        seen[digest] = filename


def build_vgg16(num_classes: int, pretrained: bool = False) -> tuple[nn.Module, object, str]:
    """Import torchvision only for the explicitly requested local-data training path."""
    try:
        from torchvision.models import VGG16_Weights, vgg16
    except Exception as error:
        raise RuntimeError("Optional training requires a working torchvision install.") from error
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = vgg16(weights=weights)
    in_features = model.classifier[0].in_features
    model.classifier = make_classifier(in_features, num_classes)
    if weights is not None:
        transform = weights.transforms()
        transform_note = "VGG16_Weights.DEFAULT.transforms()"
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        ])
        transform_note = "Resize(256) -> CenterCrop(224) -> ToTensor (no pretrained weights)"
    return model, transform, transform_note


def run_smoke() -> dict[str, object]:
    """Exercise RGB NCHW, classifier, frozen-backbone semantics, and one optimizer step."""
    torch.manual_seed(17)
    images = torch.rand(6, 3, 64, 64, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
    )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    head = make_classifier(8 * 4 * 4, 3)
    model = nn.Sequential(backbone, head)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    assert logits.shape == (6, 3) and labels.dtype == torch.long
    assert torch.isfinite(loss) and all(parameter.grad is None for parameter in backbone.parameters())
    return {"loss": float(loss.item()), "image_shape": tuple(images.shape), "logits_shape": tuple(logits.shape), "trainable_head_parameters": sum(p.numel() for p in head.parameters())}


def train_local(data_dir: str, pretrained: bool, unfreeze_last_block: bool, epochs: int, output: str) -> None:
    from torchvision.datasets import ImageFolder

    # Build the model first so pretrained weights and their transform stay a single explicit opt-in.
    dataset_probe = ImageFolder(data_dir)
    model, transform, transform_note = build_vgg16(len(dataset_probe.classes), pretrained)
    dataset = ImageFolder(data_dir, transform=transform)
    reject_duplicate_images(dataset.samples)
    train_ids, val_ids, test_ids = split_class_indices(dataset.targets)
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(17), num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_ids), batch_size=8, shuffle=False, num_workers=0)
    parameters = freeze_features(model, unfreeze_last_block)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
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
            for images, labels in val_loader:
                logits = model(images.to(device))
                correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
                total += len(labels)
        print(f"epoch {epoch + 1}/{epochs}: validation accuracy={correct / max(1, total):.3f}")
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in test_loader:
            correct += (model(images.to(device)).argmax(dim=1).cpu() == labels).sum().item()
            total += len(labels)
    artifact = {
        "model_state": model.cpu().state_dict(), "classes": dataset.classes,
        "num_classes": len(dataset.classes), "pretrained": pretrained,
        "transform": transform_note, "unfreeze_last_block": unfreeze_last_block,
        "input_shape": [3, 224, 224], "seed": 17,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    print(f"final held-out test accuracy={correct / max(1, total):.3f}; checkpoint={destination}; device={device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="local directory/class_name/image.jpg tree; no data is fetched")
    parser.add_argument("--pretrained", action="store_true", help="explicitly allow ImageNet weight download/cache access")
    parser.add_argument("--unfreeze-last-block", action="store_true", help="fine-tune the final VGG convolution block")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", default="artifacts/bs06-vgg16.pt")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.data_dir:
        train_local(args.data_dir, args.pretrained, args.unfreeze_last_block, args.epochs, args.output)
    elif args.pretrained or args.unfreeze_last_block:
        parser.error("--pretrained and --unfreeze-last-block require --data-dir; default smoke never loads VGG weights")
    else:
        result = run_smoke()
        print(f"offline synthetic transfer smoke: images={result['image_shape']}; logits={result['logits_shape']}; loss={result['loss']:.4f}; frozen backbone, trainable head={result['trainable_head_parameters']} parameters")


if __name__ == "__main__":
    main()
