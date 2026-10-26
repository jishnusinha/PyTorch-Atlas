# BS05 · Fashion-MNIST CNN

Default `python train.py` runs an offline synthetic CPU forward/backward/optimizer smoke. The two conv blocks preserve size with 3×3 same padding, then pool 28→14→7. The optional `python train.py --download-fashion-mnist --train-full --epochs 1` path explicitly downloads with torchvision. It uses the same seed-17 official train/validation partition as BS04 and the official test split. Use equal examples, metrics, and budgets for any ANN/CNN comparison; no accuracy result is assumed.

Dependencies: `torch`; optional full run: `torchvision`. Smoke test: `python -m pytest -q test_smoke.py`.
