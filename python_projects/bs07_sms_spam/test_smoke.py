from .train import SpamRNN, build_vocab, collate, run_smoke, split_rows, tokenize
import torch

def test_offline_rnn_shapes_unknown_empty_and_metrics():
    torch.set_num_threads(1)
    rows=[("known token",0),("other text",1)]
    vocab=build_vocab(rows)
    assert vocab["<pad>"]==0 and vocab["<unk>"]==1
    assert [vocab.get(t,vocab["<unk>"]) for t in tokenize("unseen word")] == [1,1]
    assert tokenize("") == []
    result=run_smoke(epochs=2)
    assert result["shape"]==(2,1) and set(result["metrics"])=={"precision","recall","f1"}
    assert torch.isfinite(torch.tensor(result["loss"]))

def test_lengths_and_checkpoint_mappings():
    result=run_smoke(epochs=1)
    assert result["vocab"]["<pad>"] != result["vocab"]["<unk>"]
    x,n,y=collate([(torch.tensor([2,3]),torch.tensor(0.)),(torch.tensor([4]),torch.tensor(1.))])
    assert x.shape==(2,2) and n.tolist()==[2,1] and y.shape==(2,1)


