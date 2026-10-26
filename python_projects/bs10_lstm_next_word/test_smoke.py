from pathlib import Path
import subprocess,sys,torch
from python_projects.bs10_lstm_next_word import train
ROOT=Path(__file__).parent

def test_special_ids_prefix_alignment_and_oov():
    torch.set_num_threads(1); tr,va=train.split_sentences(train.TINY_CORPUS,seed=42); vocab=train.build_vocab(tr)
    assert vocab[train.PAD]==0 and vocab[train.UNK]==1 and vocab[train.EOS]==2
    assert set(tr).isdisjoint(va) and train.encode("qzxwv",vocab)==[vocab[train.UNK]]
    rows=train.make_examples(["alpha beta gamma"],{train.PAD:0,train.UNK:1,train.EOS:2,"alpha":3,"beta":4,"gamma":5},4)
    assert rows==[([3],4),([3,4],5),([3,4,5],2)]

def test_training_predict_and_checkpoint_roundtrip(tmp_path):
    checkpoint=tmp_path/"model.pt"; script=ROOT/"train.py"
    fit=subprocess.run([sys.executable,str(script),"train","--epochs","1","--max-len","8","--output",str(checkpoint)],check=True,capture_output=True,text=True)
    assert "train_loss=" in fit.stdout and checkpoint.exists()
    artifact=torch.load(checkpoint,map_location="cpu",weights_only=True); assert artifact["vocab"][train.PAD]==0 and artifact["config"]["eos_id"]==2 and artifact["tokenizer"]
    pred=subprocess.run([sys.executable,str(script),"predict","--checkpoint",str(checkpoint),"--seed-text","unseen words","--max-new-tokens","3","--top-k","3","--seed","42"],check=True,capture_output=True,text=True)
    assert pred.stdout.strip().startswith("unseen words")

def test_distribution_sums_and_generation_bound():
    vocab={train.PAD:0,train.UNK:1,train.EOS:2,"hello":3}; model=train.NextWordLSTM(len(vocab),embedding_dim=4,hidden_dim=5); config={"max_len":4,"pad_id":0}
    dist=train.next_distribution(model,vocab,config,"hello",top_k=3,temperature=.8)
    assert abs(sum(prob for _,prob in dist)-1)<1e-6 and len(dist)<=3
    assert train.left_pad([3,2,3,2,3],4,0)==[2,3,2,3]
