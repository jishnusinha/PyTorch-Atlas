from .train import FashionCNN, run_smoke, split_ids


def test_cnn_shape_optimizer_and_paired_split_contract():
    result = run_smoke()
    assert result["feature_shape"] == (8, 64, 7, 7)
    assert result["output_shape"] == (8, 10)
    ann_ids, cnn_ids = split_ids(100), split_ids(100)
    assert all((a == b).all() for a, b in zip(ann_ids, cnn_ids))
    assert result["parameters"] == sum(p.numel() for p in FashionCNN().parameters())

