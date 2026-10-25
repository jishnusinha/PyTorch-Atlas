import subprocess
import sys
import torch

script = "python_examples/12_lstm_next_word/lstm_next_word.py"
train = subprocess.run([sys.executable, script, "train", "--epochs", "2",
                       "--output", "tmp/atlas-nextword.pt"], check=True, capture_output=True, text=True)
predict = subprocess.run([sys.executable, script, "predict", "--checkpoint",
                          "tmp/atlas-nextword.pt", "--seed-text", "python libraries",
                          "--max-new-tokens", "2"], check=True, capture_output=True, text=True)
assert "saved" in train.stdout and predict.stdout.strip()
print(train.stdout.splitlines()[-1], "prediction:", predict.stdout.strip())
