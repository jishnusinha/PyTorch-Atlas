# BS07 · SMS spam RNN

Run `python python_projects/bs07_sms_spam/train.py` for a deterministic CPU toy corpus. It checks train-only vocabulary, PAD/UNK IDs, sequence lengths, packed RNN states, BCEWithLogitsLoss, validation metrics, and a checkpoint containing the vocabulary. Synthetic results are not spam-filter performance claims.

Optional `--csv local.csv` expects `text,label`, with labels 0=ham and 1=spam. No URLs or downloads are used. The fixed toy split is illustrative; for real data, create a stratified held-out split before fitting the vocabulary and inspect class balance. Dependencies: PyTorch. Run `python -m pytest -q python_projects/bs07_sms_spam/test_smoke.py`.

