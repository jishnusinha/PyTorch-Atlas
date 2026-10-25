import json
import torch

record = {
    "seed": 23,
    "split": "train/validation/test = 100/40/40; test held until final report",
    "sampler": "TPE with seed 23",
    "objective": "minimize validation cross entropy",
    "trials": [
        {"number": 0, "params": {"width": 8, "lr": 0.01}, "value": 0.48, "state": "COMPLETE"},
        {"number": 1, "params": {"width": 16, "lr": 0.005}, "value": 0.41, "state": "COMPLETE"},
    ],
}
serialized = json.dumps(record, sort_keys=True)
restored = json.loads(serialized)
assert len(restored["trials"]) == 2
assert restored["split"].endswith("final report")
print(f"recorded {len(restored['trials'])} trials; config is JSON serializable")
