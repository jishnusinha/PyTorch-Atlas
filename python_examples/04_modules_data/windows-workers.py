import torch
from torch.utils.data import DataLoader,TensorDataset

def main():
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)
    y=torch.arange(8,dtype=torch.float32).reshape(8,1)
    dataset=TensorDataset(x,y)
    # Start with zero workers; test multiprocessing from a .py file, not a notebook cell.
    workers=0
    loader=DataLoader(dataset,batch_size=4,num_workers=workers)
    batches=list(loader)
    assert len(batches)==2 and batches[0][0].shape==(4,3)
    print(f"num_workers={workers}; batches={len(batches)}")

if __name__=="__main__":
    main()
