from pathlib import Path
import runpy
import torch

def test_reused_faq_end_to_end_runs_and_maps_full_answers():
    torch.set_num_threads(1)
    runpy.run_path(str(Path(__file__).parent / "train.py"), run_name="__main__")
