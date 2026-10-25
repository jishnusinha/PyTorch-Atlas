"""Run Phase 06 synthetic CPU examples. Optuna exercises are optional dependencies."""
from pathlib import Path
import runpy

import pytest
import torch

EXAMPLES = Path(__file__).parent
OPTIONAL = {"optuna-synthetic", "optuna-ann-safe"}
SCRIPTS = sorted(
    p for p in EXAMPLES.glob("*.py")
    if p.stem not in {"test_smoke", *OPTIONAL}
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_download_free_torch_example_runs(script, capsys):
    torch.set_num_threads(1)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_optional_optuna_snippets_are_explicitly_separate():
    for name in OPTIONAL:
        source = (EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
        assert "Requires optional Optuna" in source
        assert "import optuna" in source
    ann = (EXAMPLES / "optuna-ann-safe.py").read_text(encoding="utf-8")
    objective = ann.split("def objective(trial):", 1)[1].split("study =", 1)[0]
    assert "return loss" in objective
    assert "DataLoader(test" not in objective
    assert "test_data" not in objective
    assert "test = TensorDataset" in ann


def test_phase_inventory_covers_requested_concepts():
    names = {p.stem for p in SCRIPTS}
    assert len(SCRIPTS) == 8
    assert {
        "regularization-toy",
        "batchnorm-numerics",
        "weight-decay-update",
        "optimizer-comparison",
        "regularized-ann",
        "optuna-plotting",
        "study-experiments-log",
        "regularization-curves",
    } <= names
