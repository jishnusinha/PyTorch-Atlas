"""A seeded synthetic churn exercise; labels do not represent real customers."""
from __future__ import annotations

import argparse
import torch
from torch import nn


def make_data(n: int = 400):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(n, 5, generator=generator)
    score = 1.3 * x[:, 0] - 0.8 * x[:, 1] + 0.5 * x[:, 2]
    y = (score > 0.35).float().unsqueeze(1)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(23))
    return x, y, order[:int(n * .70)], order[int(n * .70):int(n * .85)], order[int(n * .85):]


def run_smoke(epochs: int = 1, evaluate_test: bool = False) -> dict[str, object]:
    torch.manual_seed(17)
    x, y, train, val, test = make_data()
    model = nn.Sequential(nn.Linear(5, 12), nn.ReLU(), nn.Linear(12, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[train]), y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_prob = model(x[val]).sigmoid()
        thresholds = torch.linspace(0.1, 0.9, 17)
        best = (-1.0, 0.5)
        for candidate in thresholds:
            predicted = val_prob >= candidate
            tp = ((predicted == 1) & (y[val] == 1)).sum().item()
            fp = ((predicted == 1) & (y[val] == 0)).sum().item()
            fn = ((predicted == 0) & (y[val] == 1)).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            if f1 > best[0]:
                best = (f1, float(candidate))
        threshold = best[1]
        val_pred = val_prob >= threshold
        test_pred = model(x[test]).sigmoid() >= threshold if evaluate_test else None
    assert val_prob.shape == (len(val), 1) and val_pred.shape == (len(val), 1)
    if test_pred is not None:
        assert test_pred.shape == (len(test), 1)
    assert set(y.unique().tolist()) == {0.0, 1.0} and torch.isfinite(loss)
    result = {"loss": loss.item(), "train": len(train), "validation": len(val), "test": len(test), "output_shape": tuple(val_pred.shape), "threshold": threshold}
    if test_pred is not None:
        truth = y[test].bool()
        tn = (~test_pred & ~truth).sum().item()
        tp = (test_pred & truth).sum().item()
        fp = (test_pred & ~truth).sum().item()
        fn = (~test_pred & truth).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        result.update({"test_precision": precision, "test_recall": recall, "test_f1": f1, "test_accuracy": (tn + tp) / len(test), "confusion": (tn, fp, fn, tp)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-full", action="store_true", help="run 100 synthetic epochs")
    args = parser.parse_args()
    result = run_smoke(epochs=100 if args.train_full else 1, evaluate_test=args.train_full)
    suffix = f"; test F1={result['test_f1']:.3f}; accuracy={result['test_accuracy']:.3f}; TN/FP/FN/TP={result['confusion']}" if args.train_full else "; test split untouched"
    print(f"synthetic churn {'full run' if args.train_full else 'smoke'}: loss={result['loss']:.4f}; output={result['output_shape']}; validation threshold={result['threshold']:.2f}; no company data used{suffix}")


if __name__ == "__main__":
    main()
