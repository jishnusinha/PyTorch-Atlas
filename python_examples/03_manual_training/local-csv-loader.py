import csv
from pathlib import Path
import torch

def read_local_csv(path):
    path=Path(path)
    if not path.is_file():
        return None
    with path.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    if not rows or "diagnosis" not in rows[0]:
        raise ValueError("CSV needs a diagnosis column")
    ignored={"id","unnamed: 32","diagnosis"}
    columns=[c for c in rows[0] if c.lower() not in ignored]
    if len(columns)!=30:
        raise ValueError(f"expected 30 feature columns, found {len(columns)}")
    mapping={"B":0.0,"M":1.0}
    labels=[r["diagnosis"].strip().upper() for r in rows]
    if any(label not in mapping for label in labels):
        raise ValueError("diagnosis must be B or M")
    x=torch.tensor([[float(r[c]) for c in columns] for r in rows],dtype=torch.float32)
    y=torch.tensor([mapping[label] for label in labels],dtype=torch.float32).reshape(-1,1)
    assert x.shape==(len(rows),30) and y.shape==(len(rows),1)
    return x,y

dataset=read_local_csv("data.csv")
if dataset is None:
    print("No local data.csv; run the synthetic preprocessing example instead.")
else:
    print("Loaded local rows:",tuple(dataset[0].shape),"labels:",tuple(dataset[1].shape))
