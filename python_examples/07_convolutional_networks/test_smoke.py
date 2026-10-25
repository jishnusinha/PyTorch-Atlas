from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).parent
EXECUTABLE = [
    "cnn-locality",
    "image-tensor-shapes",
    "manual-cross-correlation",
    "conv2d-basics",
    "output-size-formula",
    "multichannel-conv",
    "cnn-activations",
    "pooling-ops",
    "receptive-field-calc",
    "cnn-model",
    "cnn-transform",
    "cnn-shape-trace",
    "cnn-smoke-train",
    "cnn-shape-debug",
]


@pytest.mark.parametrize("name", EXECUTABLE)
def test_example_runs_on_cpu(name):
    torch.set_num_threads(1)
    runpy.run_path(str(ROOT / f"{name}.py"))


def test_optional_dataset_requires_explicit_download_flag():
    source = (ROOT / "fashionmnist-cnn-optional.py").read_text()
    assert "DOWNLOAD_DATA = False" in source
    assert "if not DOWNLOAD_DATA:" in source


def test_all_registered_phase_examples_are_mirrored():
    files = {path.stem for path in ROOT.glob("*.py")} - {"test_smoke"}
    assert files == set(EXECUTABLE) | {"fashionmnist-cnn-optional"}
