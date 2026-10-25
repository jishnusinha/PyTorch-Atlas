"""Run download-free Phase 05 examples on CPU; the optional torchvision lab is excluded."""
from pathlib import Path
import runpy

import pytest
import torch

EXAMPLES = Path(__file__).parent
SCRIPTS = sorted(
    p for p in EXAMPLES.glob("*.py")
    if p.stem not in {"fashion-mnist-optional", "test_smoke"}
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_example_runs_with_embedded_assertions(script, capsys):
    torch.set_num_threads(1)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_phase_has_download_free_examples_and_opt_in_real_data():
    assert len(SCRIPTS) == 9
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert "http://" not in source and "https://" not in source
        assert "assert " in source
    optional = (EXAMPLES / "fashion-mnist-optional.py").read_text(encoding="utf-8")
    assert "DOWNLOAD_DATA = False" in optional
    assert "download=DOWNLOAD_DATA" in optional


def test_required_ann_device_and_metric_examples_are_present():
    names = {path.stem for path in SCRIPTS}
    assert {
        "fashion-shapes",
        "ann-architecture",
        "synthetic-ann-smoke",
        "cross-entropy-contract",
        "metrics-confusion",
        "epoch-metrics",
        "device-placement",
        "checkpoint-seed",
        "seed-training-curves",
    } <= names
