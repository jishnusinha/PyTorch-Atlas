from pathlib import Path
import runpy

import torch

torch.set_num_threads(1)
root = Path(__file__).parent
def test_all_lstm_snippets_run_on_cpu():
    examples = sorted(root.glob("lstm-*.py"))
    assert len(examples) == 12, f"expected 12 LSTM snippets, found {len(examples)}"
    for example in examples:
        print(f"checking {example.name}")
        runpy.run_path(str(example), run_name="__main__")
    print(f"{len(examples)} LSTM snippets passed on CPU")
