# BS10 · LSTM next-word predictor

Reuses the audited offline CPU script in `python_examples/12_lstm_next_word/lstm_next_word.py`. The default local corpus is intentionally tiny. A sentence-level split precedes vocabulary construction; PAD, UNK, and EOS are distinct; prefixes map to the immediate next token; checkpoints include tokenizer configuration and vocabulary.

From repo root, run `python python_projects/bs10_lstm_next_word/train.py train --epochs 2 --output artifacts/bs10.pt` and then `python python_projects/bs10_lstm_next_word/train.py predict --checkpoint artifacts/bs10.pt --seed-text "python libraries" --top-k 3 --temperature 0.8`. Optional `--corpus-file local.txt` uses an explicit local file; no NLTK or remote text download. Tiny-corpus output is not useful general language modeling. Dependency: PyTorch. Test: `python -m pytest -q python_projects/bs10_lstm_next_word/test_smoke.py`.
