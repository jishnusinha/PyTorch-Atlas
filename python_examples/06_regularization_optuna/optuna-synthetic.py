"""Requires optional Optuna: python -m pip install optuna"""
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
print(f"trials={len(study.trials)}, best score={study.best_value:.3f}")
