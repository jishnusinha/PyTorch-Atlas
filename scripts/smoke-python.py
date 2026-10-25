"""Run every Phase 01 snippet; no data or weight downloads occur."""
from pathlib import Path
import runpy
import torch

torch.set_num_threads(1)
paths = sorted(Path('python_examples/01_previews').glob('*.py'))
assert len(paths) == 6
for path in paths:
    print(f'RUN {path.name}', flush=True)
    runpy.run_path(str(path), run_name='__main__')
print(f'PASS: {len(paths)} CPU snippets; torch {torch.__version__}')
