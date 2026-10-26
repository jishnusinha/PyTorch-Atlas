from .train import run_smoke


def test_builtin_dataset_pipeline_is_cpu_safe():
    result = run_smoke()
    assert result["input_features"] == 30
    assert result["output_shape"] == (85, 1)
    assert "test_f1" not in result  # smoke verifies mechanics without repeatedly opening test labels
    assert len(result["class_names"]) == 2

