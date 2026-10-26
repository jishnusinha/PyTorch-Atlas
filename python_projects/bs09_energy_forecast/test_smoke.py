from pathlib import Path
import torch
from python_projects.bs09_energy_forecast import train

def test_chronology_training_scaler_and_shapes():
    torch.set_num_threads(1); result=train.run_smoke(epochs=2); tr,va,te=result["split"]; targets=result["targets"]
    assert tr.stop==va.start and va.stop==te.start
    assert torch.allclose(result["mean"],result["series"][tr].mean(0))
    assert int(targets[targets>=va.start].min())>=va.start and int(targets[targets>=te.start].min())>=te.start
    assert result["X"].shape[1:]==(24,1) and result["Y"].shape[1:]==(1,)
    assert result["test_predictions"].shape==result["test_actual"].shape
    assert result["mae"]>=0 and result["naive_mae"]>=0

def test_window_target_follows_input_and_split_is_ordered():
    series=torch.arange(10,dtype=torch.float32).reshape(-1,1); x,y,indices=train.make_windows(series,3)
    assert x[0,:,0].tolist()==[0,1,2] and y[0,0].item()==3 and indices.tolist()==list(range(3,10))
    a,b,c=train.chronological_split(100); assert a.stop<=b.start and b.stop<=c.start


def test_local_hourly_csv_runs_forecast_and_rejects_gaps(tmp_path):
    path=tmp_path/"hourly.csv"
    lines=["timestamp,power"]+[f"2025-01-{1+i//24:02d}T{i%24:02d}:00:00,{2+i*.01}" for i in range(120)]
    path.write_text("\n".join(lines),encoding="utf-8")
    series=train.load_local_csv(path)
    result=train.run_smoke(epochs=1,series=series)
    assert result["test_predictions"].shape==result["test_actual"].shape
    lines[3]="2025-01-01T04:00:00,2.04"
    path.write_text("\n".join(lines),encoding="utf-8")
    import pytest
    with pytest.raises(ValueError,match="hourly"):
        train.load_local_csv(path)
