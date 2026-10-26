# BS02 · Wisconsin breast cancer classifier

Educational and nonclinical only. Run `python train.py` for a single CPU optimizer step on `sklearn.datasets.load_breast_cancer`; this dataset is built into scikit-learn and does not download. The example uses a stratified train/validation/test split, fits `StandardScaler` on training rows only, trains raw logits with `BCEWithLogitsLoss`, and chooses a demonstration threshold on validation rows. The smoke path never opens test labels. `python train.py --train-full` runs the longer training loop and opens the held-out test once for a final report. Dataset target names are printed so the integer encoding is not guessed.

Dependencies: `torch`, `scikit-learn`. Smoke test: `python -m pytest -q test_smoke.py`. This is not a diagnostic model.
