# BS06 · Flower classifier with VGG16

From the repository root, run `python python_projects/bs06_vgg16_flowers/train.py` for an offline CPU smoke with synthetic RGB tensors and a small feature stand-in. It verifies `[B,3,H,W]` inputs, multiclass logits, a trainable replacement head, and an optimizer step without importing torchvision or fetching weights.

For an explicitly supplied image directory, use `python python_projects/bs06_vgg16_flowers/train.py --data-dir path/to/flowers`. The tree must be `flowers/class_name/image.jpg`; install `torchvision` and Pillow. Exact duplicate file bytes are rejected before a deterministic per-class 70/15/15 split. No dataset is downloaded. Add `--pretrained` only when ImageNet weights may be fetched or loaded from cache; the selected `VGG16_Weights` object supplies its documented transform. `--unfreeze-last-block` additionally fine-tunes the final convolution block. These two controls require local `--data-dir`.

The optional VGG model uses the 512×7×7 feature grid (25,088 values) and a 128-unit replacement head. The default freezes all features; fine-tuning is an explicit second choice. The script reports validation metrics during training and uses the held-out test split once at the end. A synthetic smoke is not evidence of flower accuracy.

Dependencies: PyTorch for smoke; `torchvision` and Pillow for local images; optional ImageNet weight download only with `--pretrained`. Test with `python -m pytest -q python_projects/bs06_vgg16_flowers/test_smoke.py`.
