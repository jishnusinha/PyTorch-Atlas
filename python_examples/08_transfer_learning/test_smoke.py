from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).parent
EXECUTABLE = [
    "transfer-learning-toy",
    "vgg-feature-freeze",
    "fashion-rgb-pil",
    "vgg-transform-shapes",
    "imagenet-normalization",
    "vgg-csv-dataset",
    "vgg-shape-debug",
    "weights-transforms-api",
]


@pytest.mark.parametrize("name", EXECUTABLE)
def test_example_runs_without_downloads(name):
    torch.set_num_threads(1)
    runpy.run_path(str(ROOT / f"{name}.py"))


def test_optional_training_is_explicitly_gated():
    code = (ROOT / "vgg16-transfer-optional.py").read_text()
    assert "ALLOW_DOWNLOADS = False" in code
    assert "if not ALLOW_DOWNLOADS:" in code
    assert "weights = models.VGG16_Weights.DEFAULT" in code
    assert "validation" not in code.lower() or "val_loader" in code


def test_all_registered_phase_scripts_are_mirrored():
    files = {path.stem for path in ROOT.glob("*.py")} - {"test_smoke"}
    assert files == set(EXECUTABLE) | {"vgg16-transfer-optional"}
