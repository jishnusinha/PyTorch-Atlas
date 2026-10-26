from .train import make_data, run_smoke


def test_synthetic_split_reproducibility_and_contract():
    first = make_data()
    second = make_data()
    assert all((a == b).all() for a, b in zip(first, second))
    result = run_smoke()
    assert result["train"] == 280 and result["validation"] == 60 and result["test"] == 60

