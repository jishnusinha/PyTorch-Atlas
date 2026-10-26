from .train import run_smoke
from .train import run_local_csv


def test_regression_smoke_contract():
    result = run_smoke()
    assert result["shape"] == (14, 1)
    assert result["loss"] < result["initial_loss"]


def test_explicit_local_csv_uses_train_only_feature_scaling(tmp_path):
    path = tmp_path / "homes.csv"
    rows = ["area,bedrooms,age,distance,price"]
    rows += [f"{i},{i % 5},{i % 20},{i % 7},{3*i - 2*(i%5) + 0.2*(i%20)}" for i in range(30)]
    path.write_text("\n".join(rows), encoding="utf-8")
    result = run_local_csv(path, epochs=3)
    assert result["rows"] == 30
    assert result["test_shape"] == (5, 1)
    assert len(result["train_mean"]) == 4 and result["rmse"] >= 0

