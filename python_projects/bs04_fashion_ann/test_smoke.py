from .train import run_smoke, split_ids


def test_ann_offline_shape_loss_and_deterministic_split():
    result = run_smoke()
    assert result["input_shape"] == (64, 1, 28, 28)
    assert result["output_shape"] == (8, 10)
    first = split_ids(100)
    second = split_ids(100)
    assert all((a == b).all() for a, b in zip(first, second))
    assert sum(len(part) for part in first) == 100

