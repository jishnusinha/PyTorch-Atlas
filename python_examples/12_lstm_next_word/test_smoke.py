from pathlib import Path
import subprocess
import sys
import runpy

import torch
import pytest

ROOT = Path(__file__).parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
import lstm_next_word as project

torch.set_num_threads(1)


def test_all_copyable_phase12_snippets_run_on_cpu():
    examples = sorted(ROOT.glob("nextword-*.py"))
    assert len(examples) == 14
    for example in examples:
        print(f"checking {example.name}")
        runpy.run_path(str(example), run_name="__main__")


def test_prefix_target_alignment_vocabulary_and_sentence_split():
    train, validation = project.split_sentences(project.TINY_CORPUS, seed=42)
    assert set(train).isdisjoint(validation)
    vocab = project.build_vocab(train)
    assert vocab[project.PAD] != vocab[project.UNK]
    assert project.encode("qzxwv", vocab) == [vocab[project.UNK]]
    rows = project.make_examples(["alpha beta gamma"], {project.PAD: 0, project.UNK: 1, project.EOS: 2, "alpha": 3, "beta": 4, "gamma": 5}, max_len=4)
    assert [(x, y) for x, y in rows] == [([3], 4), ([3, 4], 5), ([3, 4, 5], 2)]


def test_cli_train_predict_and_checkpoint_round_trip(tmp_path):
    checkpoint = tmp_path / "model.pt"
    script = ROOT / "lstm_next_word.py"
    fit = subprocess.run([sys.executable, str(script), "train", "--epochs", "2", "--max-len", "8", "--output", str(checkpoint)], cwd=REPO, check=True, capture_output=True, text=True)
    assert "train_loss=" in fit.stdout and "val_loss=" in fit.stdout and checkpoint.exists()
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert artifact["vocab"][project.PAD] == 0 and artifact["vocab"][project.UNK] == 1
    assert artifact["config"]["max_len"] == 8 and artifact["model_state"]
    predicted = subprocess.run([sys.executable, str(script), "predict", "--checkpoint", str(checkpoint), "--seed-text", "unseen words", "--max-new-tokens", "3", "--top-k", "3", "--seed", "42"], cwd=REPO, check=True, capture_output=True, text=True)
    assert predicted.stdout.strip().startswith("unseen words")


def test_optional_corpus_is_an_explicit_local_file(tmp_path):
    corpus = tmp_path / "local.txt"
    corpus.write_text("red green blue\none two three\ncat dog bird\nsun moon star\n", encoding="utf-8")
    output = tmp_path / "local-model.pt"
    subprocess.run([sys.executable, str(ROOT / "lstm_next_word.py"), "train", "--corpus-file", str(corpus), "--epochs", "1", "--output", str(output)], cwd=REPO, check=True, capture_output=True, text=True)
    assert output.exists()
