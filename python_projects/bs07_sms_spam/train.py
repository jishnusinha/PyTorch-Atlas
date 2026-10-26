from __future__ import annotations
import argparse, csv, re
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
TOY_MESSAGES = [("win a free prize now",1),("claim your cash reward",1),("urgent winner click offer",1),("free entry prize claim",1),("meeting moved to tomorrow",0),("can we have lunch",0),("please send the project notes",0),("your appointment is confirmed",0),("call me when you arrive",0),("thanks for your help",0),("limited offer claim your reward",1),("team meeting starts at noon",0)]
def tokenize(text): return re.findall(r"[a-z0-9']+",text.lower())
def build_vocab(rows):
 vocab={"<pad>":0,"<unk>":1}
 for text,_ in rows:
  for token in tokenize(text):
   if token not in vocab: vocab[token]=len(vocab)
 return vocab
def split_rows(rows,seed=17):
 if len(rows)<12: raise ValueError("need at least 12 labeled messages")
 generator=torch.Generator().manual_seed(seed); train_ids=[]; val_ids=[]; test_ids=[]
 for label in (0,1):
  ids=torch.tensor([i for i,(_,y) in enumerate(rows) if y==label],dtype=torch.long)
  if len(ids)<3: raise ValueError("each class needs at least three messages for train/validation/test")
  ids=ids[torch.randperm(len(ids),generator=generator)]; n_val=max(1,int(round(.2*len(ids)))); n_test=max(1,int(round(.2*len(ids))))
  while n_val+n_test>=len(ids):
   if n_val>=n_test: n_val-=1
   else: n_test-=1
  val_ids.extend(ids[:n_val].tolist()); test_ids.extend(ids[n_val:n_val+n_test].tolist()); train_ids.extend(ids[n_val+n_test:].tolist())
 return ([rows[i] for i in sorted(train_ids)],[rows[i] for i in sorted(val_ids)],[rows[i] for i in sorted(test_ids)])

class MessageDataset(Dataset):
 def __init__(self,rows,vocab): self.rows,self.vocab=rows,vocab
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  text,label=self.rows[i]; ids=[self.vocab.get(t,self.vocab["<unk>"]) for t in tokenize(text)] or [self.vocab["<unk>"]]; return torch.tensor(ids,dtype=torch.long),torch.tensor(label,dtype=torch.float32)
def collate(batch):
 seq,target=zip(*batch); lengths=torch.tensor([len(x) for x in seq],dtype=torch.long); return pad_sequence(seq,batch_first=True,padding_value=0),lengths,torch.stack(target).reshape(-1,1)
class SpamRNN(nn.Module):
 def __init__(self,vocab_size,embedding_dim=12,hidden_size=16):
  super().__init__(); self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0); self.rnn=nn.RNN(embedding_dim,hidden_size,batch_first=True); self.head=nn.Linear(hidden_size,1)
 def forward(self,tokens,lengths):
  packed=pack_padded_sequence(self.embedding(tokens),lengths.cpu(),batch_first=True,enforce_sorted=False); _,hidden=self.rnn(packed); return self.head(hidden[-1])
def metrics(logits,targets):
 pred=logits.sigmoid()>=.5; actual=targets>=.5; tp=int((pred&actual).sum()); fp=int((pred&~actual).sum()); fn=int((~pred&actual).sum()); p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); return {"precision":p,"recall":r,"f1":2*p*r/max(1e-12,p+r)}
def load_csv(path):
 with Path(path).open(newline="",encoding="utf-8") as f: rows=[(r["text"],int(r["label"])) for r in csv.DictReader(f)]
 if len(rows)<12 or any(y not in (0,1) for _,y in rows): raise ValueError("CSV needs >=12 rows and binary labels (0=ham, 1=spam)")
 return rows
def run_smoke(epochs=12,rows=None):
 torch.manual_seed(17); torch.set_num_threads(1); rows=TOY_MESSAGES if rows is None else rows; tr,va,te=split_rows(rows); vocab=build_vocab(tr); gen=torch.Generator().manual_seed(17)
 tl=DataLoader(MessageDataset(tr,vocab),batch_size=4,shuffle=True,collate_fn=collate,generator=gen); vl=DataLoader(MessageDataset(va,vocab),batch_size=4,collate_fn=collate); test_loader=DataLoader(MessageDataset(te,vocab),batch_size=4,collate_fn=collate); model=SpamRNN(len(vocab)); opt=torch.optim.Adam(model.parameters(),lr=.03); loss_fn=nn.BCEWithLogitsLoss()
 for _ in range(epochs):
  model.train()
  for x,n,y in tl: opt.zero_grad(set_to_none=True); loss=loss_fn(model(x,n),y); loss.backward(); opt.step()
 model.eval()
 def evaluate(loader):
  logits_all=[]; targets_all=[]; losses=[]
  with torch.inference_mode():
   for xb,nb,yb in loader:
    scores=model(xb,nb); logits_all.append(scores); targets_all.append(yb); losses.append(loss_fn(scores,yb))
  scores=torch.cat(logits_all); actual=torch.cat(targets_all); return scores,actual,torch.stack(losses).mean()
 logits_val,target_val,val_loss=evaluate(vl)
 logits_test,target_test,test_loss=evaluate(test_loader)
 assert x.dtype==torch.long and n.dtype==torch.long and logits_test.shape==target_test.shape==(len(te),1) and torch.isfinite(val_loss) and torch.isfinite(test_loss) and vocab["<pad>"]!=vocab["<unk>"]
 return {"model":model,"vocab":vocab,"metrics":metrics(logits_test,target_test),"validation_metrics":metrics(logits_val,target_val),"loss":float(test_loss),"validation_loss":float(val_loss),"shape":tuple(logits_test.shape)}
def main():
 p=argparse.ArgumentParser(description="Seeded toy SMS spam RNN; local CSV explicitly opt-in."); p.add_argument("--csv"); p.add_argument("--epochs",type=int,default=12); p.add_argument("--output",default="artifacts/bs07-spam.pt"); a=p.parse_args()
 if a.epochs<1: p.error("--epochs must be positive")
 rows=load_csv(a.csv) if a.csv else TOY_MESSAGES
 o=run_smoke(a.epochs,rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({"model_state":o["model"].state_dict(),"vocab":o["vocab"],"label_map":{0:"ham",1:"spam"},"tokenizer":"lowercase word regex v1","seed":17},a.output); m=o["metrics"]; print(f"SMS run (synthetic toy labels unless --csv supplied): logits={o['shape']}; test_loss={o['loss']:.4f}; validation_F1={o['validation_metrics']['f1']:.2f}; test precision={m['precision']:.2f}, recall={m['recall']:.2f}, F1={m['f1']:.2f}; checkpoint={a.output}")
if __name__=="__main__": main()
