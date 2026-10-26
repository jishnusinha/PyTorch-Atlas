"""Educational, nonclinical classifier using scikit-learn's built-in Wisconsin dataset."""
from __future__ import annotations

import argparse
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_smoke(epochs: int = 1, evaluate_test: bool = False) -> dict[str, object]:
    torch.manual_seed(17)
    data = load_breast_cancer()
    x_train, x_hold, y_train, y_hold = train_test_split(data.data, data.target, test_size=0.30, random_state=17, stratify=data.target)
    x_val, x_test, y_val, y_test = train_test_split(x_hold, y_hold, test_size=0.50, random_state=17, stratify=y_hold)
    scaler = StandardScaler().fit(x_train)  # no held-out rows contribute statistics
    to_x = lambda array: torch.tensor(scaler.transform(array), dtype=torch.float32)
    X = to_x(x_train)
    y = torch.tensor(y_train[:, None], dtype=torch.float32)
    X_val = to_x(x_val)
    y_val_t = torch.tensor(y_val[:, None], dtype=torch.float32)
    X_test = to_x(x_test)
    model = nn.Linear(X.shape[1], 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_probability = model(X_val).sigmoid()
        # Select F1 threshold using validation predictions only; the test split stays untouched.
        thresholds = torch.linspace(0.1, 0.9, 17)
        best = (-1.0, 0.5)
        for candidate in thresholds:
            predicted = val_probability >= candidate
            tp = ((predicted == 1) & (y_val_t == 1)).sum().item()
            fp = ((predicted == 1) & (y_val_t == 0)).sum().item()
            fn = ((predicted == 0) & (y_val_t == 1)).sum().item()
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            if f1 > best[0]:
                best = (f1, float(candidate))
        threshold = best[1]
        # Smoke calls stop at validation. The held-out test is only opened for an explicit full run.
        test_probability = model(X_test).sigmoid() if evaluate_test else None
    assert val_probability.shape == y_val_t.shape
    if test_probability is not None:
        prediction = test_probability >= threshold
        assert prediction.shape == (len(y_test), 1)
    assert y.dtype == torch.float32 and torch.isfinite(loss)
    result = {"loss": loss.item(), "input_features": X.shape[1], "val_size": len(y_val), "test_size": len(y_test), "output_shape": tuple(val_probability.shape), "class_names": data.target_names.tolist(), "threshold": threshold}
    if test_probability is not None:
        truth = torch.tensor(y_test[:, None], dtype=torch.bool)
        tn = (~prediction & ~truth).sum().item()
        tp = (prediction & truth).sum().item()
        fp = (prediction & ~truth).sum().item()
        fn = (~prediction & truth).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        result.update({"test_precision": precision, "test_recall": recall, "test_f1": f1, "test_accuracy": (tn + tp) / len(y_test), "confusion": (tn, fp, fn, tp)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-full", action="store_true", help="repeat the training update for 100 epochs")
    args = parser.parse_args()
    # The default is one fast CPU step; the longer built-in-data run is explicit.
    result = run_smoke(epochs=100 if args.train_full else 1, evaluate_test=args.train_full)
    suffix = f"; test F1={result['test_f1']:.3f}; accuracy={result['test_accuracy']:.3f}; TN/FP/FN/TP={result['confusion']}" if args.train_full else "; test split untouched"
    print(f"nonclinical dataset {'full' if args.train_full else 'smoke'}: features={result['input_features']}; loss={result['loss']:.4f}; validation output={result['output_shape']}; classes={result['class_names']}; validation threshold={result['threshold']:.2f}{suffix}")


if __name__ == "__main__":
    main()
