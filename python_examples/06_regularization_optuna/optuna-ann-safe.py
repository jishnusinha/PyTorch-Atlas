"""Requires optional Optuna: python -m pip install optuna"""
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
print(f"best validation loss={best_loss:.3f}; one final test accuracy={correct / count:.3f}")
