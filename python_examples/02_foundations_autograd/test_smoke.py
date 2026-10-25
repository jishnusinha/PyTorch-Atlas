"""CPU smoke tests for every Phase 02 example; samples contain no downloads."""
from pathlib import Path
import runpy

import pytest
import torch

EXAMPLES = Path(__file__).parent
SCRIPTS = sorted(p for p in EXAMPLES.glob("*.py") if p.name != Path(__file__).name)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_standalone_example_runs_with_assertions(script, capsys):
    torch.set_num_threads(1)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_phase_has_twelve_download_free_examples():
    assert len(SCRIPTS) == 14
    assert all("download" not in path.read_text(encoding="utf-8").lower() for path in SCRIPTS)
