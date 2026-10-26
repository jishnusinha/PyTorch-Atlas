"""Seeded, offline regression example. Outputs are teaching-run values, not market results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import torch
from torch import nn


def make_data(n: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(n, 4, generator=generator)
    # A known affine rule makes training behavior easy to inspect.
    y = 3.0 * x[:, :1] - 1.5 * x[:, 1:2] + 0.4
    return x.float(), y.float()


def run_smoke() -> dict[str, object]:
    torch.manual_seed(17)
    x, y = make_data()
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(23))
    train, val, test = order[:67], order[67:82], order[82:]
    baseline = y[train].mean()
    assert torch.isfinite((y[val] - baseline).abs().mean())
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        first = loss_fn(model(x[train]), y[train]).item()
    for _ in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[train]), y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(x[test])
        residual = prediction - y[test]
        mae = residual.abs().mean().item()
        rmse = residual.square().mean().sqrt().item()
    assert prediction.shape == (len(test), 1)
    assert torch.isfinite(loss) and loss.item() < first
    return {"loss": loss.item(), "initial_loss": first, "mae": mae, "rmse": rmse, "shape": tuple(prediction.shape), "val_size": len(val)}


def run_local_csv(path: str | Path, epochs: int = 200) -> dict[str, object]:
    """Train on an explicitly selected CSV with area, bedrooms, age, distance, price columns."""
    required = ("area", "bedrooms", "age", "distance", "price")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain these columns: {', '.join(required)}")
        rows = [[float(row[name]) for name in required] for row in reader]
    if len(rows) < 20 or not torch.isfinite(torch.tensor(rows)).all():
        raise ValueError("Provide at least 20 complete finite CSV rows.")
    values = torch.tensor(rows, dtype=torch.float32)
    x, y = values[:, :4], values[:, 4:5]
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(17))
    train_end, val_end = int(len(x) * .70), int(len(x) * .85)
    train, val, test = order[:train_end], order[train_end:val_end], order[val_end:]
    mean = x[train].mean(dim=0)
    scale = x[train].std(dim=0, unbiased=False).clamp_min(1e-6)
    # Reuse training-only statistics for validation, test and later inference.
    x = (x - mean) / scale
    model = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[train]), y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_prediction = model(x[val])
        prediction = model(x[test])
        residual = prediction - y[test]
    return {"rows":len(rows),"val_shape":tuple(val_prediction.shape),"test_shape":tuple(prediction.shape),"mae":residual.abs().mean().item(),"rmse":residual.square().mean().sqrt().item(),"train_mean":mean.tolist(),"train_scale":scale.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-full", action="store_true", help="use a larger synthetic teaching run")
    parser.add_argument("--csv", type=Path, help="explicit local CSV with area,bedrooms,age,distance,price columns")
    args = parser.parse_args()
    if args.csv:
        result = run_local_csv(args.csv)
        print(f"local CSV run: rows={result['rows']}; held-out MAE={result['mae']:.4f}; RMSE={result['rmse']:.4f}; scaler fit on train rows only")
    elif args.train_full:
        # Explicitly longer CPU run, still synthetic and not a housing benchmark.
        torch.manual_seed(17)
        x, y = make_data(2048)
        model = nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        loss_fn = nn.MSELoss()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"synthetic full-run MSE: {loss.item():.5f} (not a market result)")
    else:
        result = run_smoke()
        print(f"synthetic smoke: loss={result['loss']:.5f}; held-out MAE={result['mae']:.4f}; output={result['shape']}")


if __name__ == "__main__":
    main()
