"""CPU-first, offline next-token LSTM. No corpus is downloaded."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
TINY_CORPUS = """python libraries help people build useful tools
small models learn patterns from examples
language models predict the next token in context
careful tests check shapes and training behavior
recurrent networks process tokens in sequence
embeddings map token ids to dense vectors
the course follows a monthly subscription plan
data science uses python libraries and clear reasoning
"""


def tokenize(text: str) -> list[str]:
    """A documented local tokenizer; keeps words and punctuation, no NLTK data."""
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower(), flags=re.UNICODE)


def split_sentences(text: str, seed: int, validation_fraction: float = 0.25):
    sentences = [line.strip() for line in text.splitlines() if tokenize(line.strip())]
    if len(sentences) < 4:
        raise ValueError("Provide at least four non-empty lines for a sentence-level split.")
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    n_val = min(len(sentences) - 1, max(1, round(len(sentences) * validation_fraction)))
    val_ids = set(order[:n_val])
    train = [s for i, s in enumerate(sentences) if i not in val_ids]
    validation = [s for i, s in enumerate(sentences) if i in val_ids]
    return train, validation


def build_vocab(sentences: Iterable[str]) -> dict[str, int]:
    counts = Counter(token for sentence in sentences for token in tokenize(sentence))
    vocab = {PAD: 0, UNK: 1, EOS: 2}
    # Stable tie-break means the saved mapping is reproducible across runs.
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(token, vocab[UNK]) for token in tokenize(text)]


def make_examples(sentences: Iterable[str], vocab: dict[str, int], max_len: int):
    """Create causal prefixes; y is always the token immediately after x."""
    rows = []
    for sentence in sentences:
        ids = encode(sentence, vocab) + [vocab[EOS]]
        for end in range(1, len(ids)):
            prefix, target = ids[:end], ids[end]
            prefix = prefix[-max_len:]
            rows.append((prefix, target))
    return rows


def left_pad(prefix: list[int], width: int, pad_id: int) -> list[int]:
    # Retained for source fidelity: the PDF prepends zeros, then takes the last item as y.
    prefix = prefix[-width:]
    return [pad_id] * (width - len(prefix)) + prefix


class PrefixDataset(Dataset):
    def __init__(self, rows, width: int, pad_id: int):
        self.rows, self.width, self.pad_id = rows, width, pad_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        prefix, target = self.rows[index]
        x = torch.tensor(left_pad(prefix, self.width, self.pad_id), dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y


class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 24, hidden_dim: int = 32, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, (h_n, _c_n) = self.lstm(embedded)
        return self.head(h_n[-1])


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for x, y in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / max(1, count)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    corpus = Path(args.corpus_file).read_text(encoding="utf-8") if args.corpus_file else TINY_CORPUS
    train_sentences, val_sentences = split_sentences(corpus, args.seed)
    vocab = build_vocab(train_sentences)  # validation text never defines the mapping
    max_len = args.max_len
    train_rows = make_examples(train_sentences, vocab, max_len)
    val_rows = make_examples(val_sentences, vocab, max_len)
    train_set = PrefixDataset(train_rows, max_len, vocab[PAD])
    val_set = PrefixDataset(val_rows, max_len, vocab[PAD])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    config = {"vocab_size": len(vocab), "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
              "max_len": max_len, "pad_id": vocab[PAD], "unk_id": vocab[UNK], "eos_id": vocab[EOS]}
    model = NextWordLSTM(**{k: config[k] for k in ("vocab_size", "embedding_dim", "hidden_dim", "pad_id")})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        with torch.inference_mode():
            val_loss = run_epoch(model, val_loader, criterion)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    artifact = {"model_state": model.state_dict(), "vocab": vocab, "config": config,
                "seed": args.seed, "tokenizer": "lowercase word/punctuation regex v1"}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, destination)
    print(f"saved {destination} · vocab={len(vocab)} · train prefixes={len(train_set)} · validation prefixes={len(val_set)}")


def load_model(path: str):
    # weights_only=True is safe for this tensor + primitive checkpoint structure.
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    config, vocab = artifact["config"], artifact["vocab"]
    model = NextWordLSTM(**{k: config[k] for k in ("vocab_size", "embedding_dim", "hidden_dim", "pad_id")})
    model.load_state_dict(artifact["model_state"])
    model.eval()
    return model, vocab, config


def next_distribution(model, vocab, config, text: str, top_k: int = 1, temperature: float = 1.0):
    if temperature <= 0 or top_k < 1:
        raise ValueError("temperature must be positive and top_k must be at least 1")
    ids = encode(text, vocab)
    ids = left_pad(ids, config["max_len"], config["pad_id"])
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits = model(x)[0]
        logits[config["pad_id"]] = -torch.inf
        scaled = logits / temperature
        k = min(top_k, len(vocab) - 1)
        values, indices = torch.topk(scaled, k=k)
        probabilities = torch.softmax(values, dim=0)
    return [(vocab_token(vocab, idx.item()), float(prob)) for idx, prob in zip(indices, probabilities)]


def vocab_token(vocab, index: int):
    return next(token for token, value in vocab.items() if value == index)


def predict(args):
    model, vocab, config = load_model(args.checkpoint)
    current = args.seed_text.strip()
    random_gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.max_new_tokens):
        candidates = next_distribution(model, vocab, config, current, args.top_k, args.temperature)
        if args.top_k == 1:
            token = candidates[0][0]
        else:
            probs = torch.tensor([p for _, p in candidates], dtype=torch.float32)
            selected = torch.multinomial(probs, 1, generator=random_gen).item()
            token = candidates[selected][0]
        if token == EOS:
            break
        current += (" " if current else "") + token
        if token == UNK:
            # Keep the unknown marker visible; the model has no word-level meaning for it.
            break
    print(current)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="train on the inline or a user-provided local corpus")
    fit.add_argument("--corpus-file", help="optional local UTF-8 text, one sentence per line; no download")
    fit.add_argument("--output", default="artifacts/nextword.pt")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--max-len", type=int, default=16)
    fit.add_argument("--embedding-dim", type=int, default=24)
    fit.add_argument("--hidden-dim", type=int, default=32)
    fit.add_argument("--learning-rate", type=float, default=0.001)
    fit.set_defaults(func=train)
    infer = commands.add_parser("predict", help="greedy or top-k temperature next-token generation")
    infer.add_argument("--checkpoint", default="artifacts/nextword.pt")
    infer.add_argument("--seed-text", default="python libraries")
    infer.add_argument("--max-new-tokens", type=int, default=8)
    infer.add_argument("--top-k", type=int, default=1, help="1 selects greedy argmax; >1 samples within top-k")
    infer.add_argument("--temperature", type=float, default=1.0)
    infer.add_argument("--seed", type=int, default=42)
    infer.set_defaults(func=predict)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
