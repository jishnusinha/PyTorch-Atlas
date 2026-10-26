# BS03 · Synthetic customer churn

The default seeded labels follow a small synthetic rule. They are not real customer data or representative churn statistics. Run `python train.py` for a CPU smoke, or `python train.py --train-full` for a longer synthetic run. Compare with a logistic baseline before treating an ANN as useful. Any Optuna extension must score validation data, never test labels.

Dependencies: `torch`. Smoke test: `python -m pytest -q test_smoke.py`.
