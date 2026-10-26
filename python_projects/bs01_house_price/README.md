# BS01 · House price regression

Run `python train.py` for the seeded offline CPU smoke, `python train.py --train-full` for a longer synthetic exercise, or `python train.py --csv path/to/homes.csv` for an explicitly selected local file. The CSV mode expects `area,bedrooms,age,distance,price`, requires 20 finite rows, and fits feature scaling on training rows only. No real housing rows are bundled or downloaded; document units and missing-value policy before using a local file.

Dependencies: `torch`. Smoke test: `python -m pytest -q test_smoke.py`.
