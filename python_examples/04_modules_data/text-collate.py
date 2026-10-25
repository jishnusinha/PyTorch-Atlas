import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset

class Sentences(Dataset):
    def __init__(self,rows):
        self.rows=rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        words,label=self.rows[index]
        return torch.tensor(words,dtype=torch.long),label
def collate_text(items):
    sequences,labels=zip(*items)
    padded=pad_sequence(sequences,batch_first=True,padding_value=0)
    lengths=torch.tensor([len(s) for s in sequences],dtype=torch.long)
    label_ids={"negative":0,"positive":1}
    targets=torch.tensor([label_ids[label] for label in labels],dtype=torch.long)
    return padded,lengths,targets
rows=[([4,5],"positive"),([7,8,9,10],"negative"),([2,3,4],"positive")]
tokens,lengths,labels=next(iter(DataLoader(Sentences(rows),batch_size=3,collate_fn=collate_text)))
assert tokens.shape==(3,4) and lengths.tolist()==[2,4,3]
assert labels.tolist()==[1,0,1] and tokens[0,2:].tolist()==[0,0]
print("padded tokens",tuple(tokens.shape),"lengths",lengths.tolist())
