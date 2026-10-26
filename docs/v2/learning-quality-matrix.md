# F15 learning-quality matrix

Baseline: `origin/main` at `55692fd` (F14 merged). The reader currently has 94 stable lesson IDs and 236 lesson snippet slots backed by 135 unique snippet IDs. A scan of the canonical `lines` arrays found 131 slots across 65 unique snippets that contain generic fallback wording such as “Trace this step through …” or “Apply this statement …”. These notes describe neither a particular operation nor its tensor contract. Hand-authored explanations in the preview, phase 02, and parts of phases 10–12 are already specific and should be retained.

## Priority snippet annotations

The following 33 canonical snippets are selected for typed, line-linked WHY/shape/gotcha annotations. Explanations will live in the reader and will not alter the code users copy or download.

| Area | Snippets |
|---|---|
| Tensor and autograd | `tensor-shapes`, `autograd-accumulate-clear`, `autograd-numeric-check` |
| Modules, losses and data | `manual-vs-idiomatic`, `module-hooks`, `state-dict-modes`, `optimizer-lifecycle`, `loss-shapes`, `manual-bce`, `cross-entropy-contract`, `dataloader-pipeline`, `custom-dataset` |
| Optuna | `optuna-ann-safe`, `study-experiments-log` |
| CNN | `cnn-shape-trace`, `cnn-shape-debug`, `output-size-formula` |
| Transfer learning | `vgg-feature-freeze`, `weights-transforms-api`, `vgg-shape-debug` |
| RNN and gradient flow | `rnn-padding-pack`, `gradient-clipping`, `bptt-shared-gradients` |
| FAQ classifier | `faq-checkpoint`, `faq-loss-targets`, `faq-oov` |
| LSTM | `lstm-parameter-count`, `lstm-cell-manual`, `lstm-shape-basic` |
| Next-word model | `nextword-sampling`, `nextword-prefix-pairs`, `nextword-padding`, `nextword-train-step` |

## Lesson improvements

All changes below add a compact worked check, an explicit tensor path, and a direct link to the runnable canonical example. Existing lesson IDs, source spans, diagrams, and authored prose stay intact. Comparison callouts will be attached to the relevant existing lessons and will distinguish source implementation from engineering repair.

| Lesson | Source pages reviewed | Specific learner check / shape path | Practice link |
|---|---|---|---|
| `tensor-prerequisites` | PDF pp. 1–1 | `[2,3] @ [3,2] → [2,2]`; name batch and feature axes | `tensor-shapes` |
| `scalar-autograd` | PDF pp. 3–3 | `x=3 → x²=9 → dx-gradient=6`; scalar leaf and scalar derivative | `autograd-numeric-check` |
| `vector-gradients` | PDF pp. 4–4 | Mean reduction over three values; explain the `1/3` gradient factor | `autograd-accumulate-clear` |
| `manual-loss` | PDF pp. 12–13 | `w=.2`, gradient `.4`, learning rate `.05` gives `.18` | `manual-bce` |
| `manual-training-loop` | PDF pp. 13–14 | Trace one sample through `[1,30] → [1,1]` logit and binary target | `full-manual-classifier` |
| `loss-conventions` | PDF pp. 18–20 | `z=2,y=1`: sigmoid probability and stable BCE loss; preserve logits | `loss-shapes` |
| `optimizer-lifecycle` | PDF pp. 21–22 | One scalar update and the zero-grad/backward/step ordering | `optimizer-lifecycle` |
| `dataloader-pipeline` | PDF pp. 29–36 | `N=10,B=4 → 4+4+2`; show which samples `drop_last` omits | `dataloader-pipeline` |
| `preprocessing` | PDF pp. 6–11 | Fit mean/scale on train rows, then reuse for validation/test | `manual-preprocessing` |
| `ann-evaluation` | PDF pp. 45–50 | Read a small confusion matrix by true-label row and predicted column | `metrics-confusion` |
| `ann-objective` | PDF pp. 81–87 | Separate train updates, validation trial score, and final test report | `optuna-ann-safe` |
| `cnn-classifier` | PDF pp. 89–97, 119–121 | `[2,1,28,28] → [2,32,14,14] → [2,64,7,7] → [2,10]` | `cnn-shape-trace` |
| `convolution-shapes` | PDF pp. 105–108 | `28×28`, `K=3`, `P=1`, `S=2` gives `14×14` | `output-size-formula` |
| `vgg16-lab` | PDF pp. 128–141 | `[B,3,224,224] → [B,512,7,7] → [B,25088] → [B,10]` | `vgg-feature-freeze` |
| `rnn-shapes` | PDF pp. 148–151 | `[B=2,T=4,F=3] → output [2,4,5]`, `h_n [1,2,5]` | `rnn-padding-pack` |
| `bptt` | PDF pp. 158–162 | Show a length-three loss path and sum of shared-weight gradient contributions | `bptt-shared-gradients` |
| `gradient-pathologies` | PDF pp. 163–167 | Compare `0.5⁴=.0625` with clipping’s distinct role | `gradient-clipping` |
| `faq-classifier` | PDF pp. 170–179 | `[1,T] → [1,C] → class ID → stored full response` | `faq-loss-targets` |
| `lstm-gates` | PDF pp. 188–198 | One coordinate: `c=.5×4+.25×.8=2.2`; with `o=.6`, `h=.6×tanh(2.2)≈.585` | `lstm-cell-manual` |
| `next-word-vocabulary` | PDF pp. 200–207 | Distinguish PAD/UNK/EOS IDs and trace `[4,8] → target 9` | `nextword-prefix-pairs` |
| `next-word-training` | PDF pp. 208–210 | `[B,T] long → embedding [B,T,E] → logits [B,V]`, target `[B] long` | `nextword-train-step` |

## Source comparisons

| Comparison | Source basis | Reader placement |
|---|---|---|
| Manual parameter update vs `nn.Module` and optimizer | PDF pp. 12–14, 17–22 | `manual-vs-idiomatic`, `optimizer-lifecycle` |
| `Sigmoid + BCELoss` vs `BCEWithLogitsLoss` | PDF pp. 12–13, 18–20 | `manual-loss`, `loss-conventions` |
| Source Optuna test-score selection vs validation selection | PDF pp. 81–87; source audit repair | `ann-objective` |
| FAQ next-answer-token vs answer-class to stored full response | PDF pp. 170–179; engineering repair | `faq-classifier` |
| Shared PAD/UNK ID vs distinct PAD/UNK/EOS | PDF pp. 201–202; engineering repair | `next-word-vocabulary` |
| Source Keras single-bias count vs PyTorch’s two recurrent biases; native LSTM layer/direction count | PDF pp. 150–151; LSTM formula is a supplemental framework comparison | `rnn-parameter-count`, `lstm-parameters` |
