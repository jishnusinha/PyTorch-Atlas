"""Fashion-MNIST ANN with offline synthetic smoke and opt-in torchvision download."""
from __future__ import annotations

import argparse
from functools import wraps
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def single_threaded_cpu(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return function(*args, **kwargs)
        finally:
            torch.set_num_threads(previous)
    return wrapped


def make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))


def split_ids(n: int, seed: int = 17) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_end, val_end = int(n * .70), int(n * .85)
    return order[:train_end], order[train_end:val_end], order[val_end:]


@single_threaded_cpu
def run_smoke() -> dict[str, object]:
    torch.manual_seed(17)
    x = torch.rand(64, 1, 28, 28)
    y = torch.arange(64, dtype=torch.long) % 10
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(x[:8])
    assert logits.shape == (64, 10) and output.shape == (8, 10)
    assert y.dtype == torch.long and torch.isfinite(loss)
    return {"loss": loss.item(), "input_shape": tuple(x.shape), "output_shape": tuple(output.shape), "split_ids": split_ids(len(x))}


def train_downloaded(epochs: int) -> None:
    # This branch is entered only when both explicit CLI flags are supplied.
    from torchvision import datasets, transforms
    from torch.utils.data import Subset
    root = "./data"
    transform = transforms.ToTensor()
    train_all = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    train_ids, val_ids, _ = split_ids(len(train_all))
    train_set, val_set = Subset(train_all, train_ids.tolist()), Subset(train_all, val_ids.tolist())
    generator = torch.Generator().manual_seed(17)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    torch.manual_seed(17)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train(); total = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step(); total += loss.item()
        model.eval()
        with torch.no_grad():
            correct = sum((model(images).argmax(1) == labels).sum().item() for images, labels in val_loader)
        print(f"epoch {epoch+1}: mean train loss={total/len(train_loader):.4f}; validation accuracy={correct/len(val_set):.4f}")
    model.eval()
    with torch.no_grad():
        correct = sum((model(images).argmax(1) == labels).sum().item() for images, labels in test_loader)
    print(f"held-out test accuracy={correct/len(test_set):.4f} (measured by this run; not a benchmark claim)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-fashion-mnist", action="store_true", help="allow torchvision to download official data")
    parser.add_argument("--train-full", action="store_true", help="train on downloaded data; requires the explicit download flag")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.train_full and not args.download_fashion_mnist:
        parser.error("--train-full requires --download-fashion-mnist")
    if args.download_fashion_mnist:
        if not args.train_full:
            parser.error("download mode also requires --train-full; default smoke never downloads")
        train_downloaded(args.epochs)
    else:
        result = run_smoke()
        print(f"synthetic ANN smoke: {result['input_shape']} -> {result['output_shape']}; finite loss={result['loss']:.4f}; no data download")


if __name__ == "__main__":
    main()
