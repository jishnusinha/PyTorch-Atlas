"""Run all self-contained Phase 04 examples under CPU PyTorch."""
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


def test_phase_has_at_least_twelve_download_free_examples():
    assert len(SCRIPTS) >= 12
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert "http://" not in source and "https://" not in source
        assert "assert " in source


def test_required_model_data_and_checkpoint_examples_are_present():
    names = {path.stem for path in SCRIPTS}
    assert {
        "idiomatic-classifier",
        "custom-dataset",
        "tensor-dataset",
        "dataloader-pipeline",
        "text-collate",
        "state-dict-modes",
        "batched-classifier",
        "windows-workers",
    } <= names
