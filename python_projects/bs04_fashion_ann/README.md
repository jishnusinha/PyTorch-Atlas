# BS04 · Fashion-MNIST ANN

Default `python train.py` runs a seeded synthetic CPU shape/loss smoke and never imports torchvision or downloads data. For the optional real dataset run, explicitly call `python train.py --download-fashion-mnist --train-full --epochs 1`; this downloads through torchvision into `./data`. The training and validation rows are selected with seed 17. The official test split is evaluated after training. Run count is chosen by you; metrics are specific to that run and are not bundled as expected benchmark values.

Dependencies: `torch`; optional full run: `torchvision`. Smoke test: `python -m pytest -q test_smoke.py`.
