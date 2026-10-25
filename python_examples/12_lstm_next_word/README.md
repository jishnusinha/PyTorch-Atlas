# Local next-word LSTM

Requires Python 3.10+ and PyTorch. The default corpus is inline and training runs on CPU; the script does not download data or require NLTK, a GPU, credentials, or a service.

From the repository root:

```powershell
python python_examples/12_lstm_next_word/lstm_next_word.py train
python python_examples/12_lstm_next_word/lstm_next_word.py predict --seed-text "python libraries" --max-new-tokens 8
python -m pytest -q python_examples/12_lstm_next_word/test_smoke.py
```

Training writes `artifacts/nextword.pt` with a state dict, token mapping, special IDs, model dimensions and tokenizer version. To use a larger corpus, provide a local UTF-8 text file with one sentence per line using `train --corpus-file path/to/text.txt`; inspect its licensing, privacy and train/validation grouping before use. No dataset is fetched automatically.

The tiny corpus validates mechanics only. Its losses and generated phrases are not quality benchmarks.
