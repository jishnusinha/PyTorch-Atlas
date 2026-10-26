from __future__ import annotations
import argparse,csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

def make_series(n=24*28,seed=17):
    t=torch.arange(n,dtype=torch.float32); gen=torch.Generator().manual_seed(seed); noise=.08*torch.randn(n,generator=gen)
    daily=torch.sin(2*torch.pi*t/24); weekly=.35*torch.sin(2*torch.pi*t/(24*7)); trend=.0008*t
    return (2.0+daily+weekly+trend+noise).reshape(-1,1)
def chronological_split(n):
    a=int(.70*n); b=int(.85*n); return slice(0,a),slice(a,b),slice(b,n)
def fit_scaler(series,train_slice):
    train=series[train_slice]; mean=train.mean(0); scale=train.std(0,unbiased=False).clamp_min(1e-6); return mean,scale
def make_windows(scaled,window):
    x=[]; y=[]; target_indices=[]
    for target in range(window,len(scaled)):
        x.append(scaled[target-window:target]); y.append(scaled[target]); target_indices.append(target)
    return torch.stack(x),torch.stack(y),torch.tensor(target_indices,dtype=torch.long)
def make_model(window=24):
    class ForecastLSTM(nn.Module):
        def __init__(self):
            super().__init__(); self.lstm=nn.LSTM(1,16,batch_first=True); self.head=nn.Linear(16,1)
        def forward(self,x):
            _,(hidden,_)=self.lstm(x); return self.head(hidden[-1])
    return ForecastLSTM()
def run_smoke(epochs=12,series=None):
    torch.manual_seed(17); torch.set_num_threads(1); series=make_series() if series is None else series.float().reshape(-1,1); train,val,test=chronological_split(len(series)); mean,scale=fit_scaler(series,train); z=(series-mean)/scale
    window=24; X,Y,targets=make_windows(z,window); train_ids=(targets<train.stop).nonzero().flatten(); val_ids=((targets>=val.start)&(targets<val.stop)).nonzero().flatten(); test_ids=(targets>=test.start).nonzero().flatten()
    assert len(train_ids) and len(val_ids) and len(test_ids) and int(targets[val_ids].min())>=val.start and int(targets[test_ids].min())>=test.start
    loader=DataLoader(TensorDataset(X[train_ids],Y[train_ids]),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(17)); model=make_model(); optimizer=torch.optim.Adam(model.parameters(),lr=.01); loss_fn=nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in loader:
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode(): pred_z=model(X[test_ids]); truth_z=Y[test_ids]; naive_z=X[test_ids, -1, :]
    pred=pred_z*scale+mean; truth=truth_z*scale+mean; naive=naive_z*scale+mean
    mae=(pred-truth).abs().mean(); naive_mae=(naive-truth).abs().mean(); rmse=(pred-truth).square().mean().sqrt()
    assert X.shape[1:]==(window,1) and Y.shape[1:]==(1,) and pred.shape==truth.shape and torch.isfinite(mae)
    return {"series":series,"X":X,"Y":Y,"targets":targets,"split":(train,val,test),"mean":mean,"scale":scale,"mae":float(mae),"rmse":float(rmse),"naive_mae":float(naive_mae),"test_predictions":pred,"test_actual":truth}
def load_local_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows or not {"timestamp","power"}.issubset(rows[0]): raise ValueError("CSV requires timestamp,power columns")
    from datetime import datetime
    rows.sort(key=lambda r:r["timestamp"]); values=torch.tensor([[float(r["power"])] for r in rows],dtype=torch.float32)
    if len(values)<100 or not torch.isfinite(values).all(): raise ValueError("provide >=100 finite hourly rows; sort and verify cadence")
    stamps=[datetime.fromisoformat(r["timestamp"].replace("Z","+00:00")) for r in rows]
    if any((right-left).total_seconds()!=3600 for left,right in zip(stamps,stamps[1:])): raise ValueError("timestamps must be unique, increasing hourly observations with no gaps")
    return values

def main():
    p=argparse.ArgumentParser(description="CPU synthetic hourly energy LSTM; no data downloads."); p.add_argument("--csv",help="optional local timestamp,power CSV"); p.add_argument("--epochs",type=int,default=12); a=p.parse_args()
    if a.epochs<1:p.error("--epochs must be positive")
    is_local=bool(a.csv); s=load_local_csv(a.csv) if a.csv else None
    r=run_smoke(a.epochs,series=s); label="local CSV" if is_local else "synthetic hourly demo (not a real energy benchmark)"
    print(f"{label}: X={tuple(r['X'].shape)} y={tuple(r['Y'].shape)} test MAE={r['mae']:.3f}, RMSE={r['rmse']:.3f}, persistence MAE={r['naive_mae']:.3f}")
if __name__=="__main__":main()
