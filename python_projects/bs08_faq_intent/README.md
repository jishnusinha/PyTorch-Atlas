# BS08 · FAQ intent classifier

This project reuses the reconstructed PDF-derived FAQ classifier from `python_examples/10_faq_classifier/faq-end-to-end.py`. It trains a small finite intent model on a synthetic set, stores full approved answer strings in the checkpoint, and demonstrates an answer lookup. It is a toy intent router, not general question answering or a generative assistant.

Run from repository root: `python python_projects/bs08_faq_intent/train.py`. Dependencies: PyTorch only. No download. Tiny paraphrase splits do not establish broad intent coverage; softmax confidence is not proof an input is supported. For production, validate an abstention policy on representative out-of-domain questions.
