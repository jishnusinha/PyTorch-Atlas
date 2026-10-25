"""Execute each standalone Phase 03 example and assert training behavior."""
from pathlib import Path
import runpy

import pytest
import torch

EXAMPLES = Path(__file__).parent
SCRIPTS = sorted(p for p in EXAMPLES.glob("*.py") if p.name != Path(__file__).name)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_example_runs_with_embedded_assertions(script, capsys):
    torch.set_num_threads(1)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_phase_has_eight_download_free_examples():
    assert len(SCRIPTS) == 8
    assert all("download" not in path.read_text(encoding="utf-8").lower() for path in SCRIPTS)


def test_full_classifier_script_asserts_training_and_shape_invariants():
    source = (EXAMPLES / "full-manual-classifier.py").read_text(encoding="utf-8")
    assert "final.item()<initial" in source
    assert "accuracy.item()>.75" in source
    assert "model.weights.shape==(30,1)" in source
    assert "torch.float32" in source
