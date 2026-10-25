from pathlib import Path
import runpy

import pytest
import torch

ROOT = Path(__file__).parent
EXECUTABLE = [
    "rnn-batch-shape",
    "rnn-parameter-count",
    "rnn-cell-manual",
    "rnn-many-to-one",
    "rnn-many-to-many",
    "rnn-padding-pack",
    "bptt-shared-gradients",
    "gradient-products",
    "gradient-clipping",
    "rnn-train-eval",
]


@pytest.mark.parametrize("name", EXECUTABLE)
def test_example_runs_on_cpu_without_downloads(name):
    torch.set_num_threads(1)
    runpy.run_path(str(ROOT / f"{name}.py"))


def test_all_phase_scripts_are_registered_and_mirrored():
    files = {path.stem for path in ROOT.glob("*.py")} - {"test_smoke"}
    assert files == set(EXECUTABLE)
