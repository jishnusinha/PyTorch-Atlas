# Source audit — Phase 01

All 211 pages were extracted in order using PDF.js. Full page text is retained in `docs/source-pages/`; it is audit material and is never imported by the application. Each page has an extraction character count in `pdf-page-index.json`. The export repeats its main text layer twice on most pages; `scripts/read-pages.mjs` removes the duplicate for reading, leaving the preserved extracts untouched. Mathematical notation is fragmented by layout, so equations must be reconstructed and numerically checked. This is a source index, not a claim of completed lesson coverage.

The 16 source blocks preserve the prompt pack's deliberate overlapping boundaries. Topic spans refine these boundaries: ANN evaluation finishes on p.50; GPU evaluation finishes on p.60; CNN verification and transfer introduction share p.121; LSTM discussion and predictor introduction share p.198.

## Repairs to carry into later phases

- pp.11, 21, 30, 36: dataset URLs and other long code lines are visibly clipped in the export. The source itself records an HTTP failure on pp.35–36. Use local files or explicit optional downloads; do not restore URLs by guessing.
- pp.7, 12: manual code uses float64; the module implementation correctly switches to float32 on p.22. Explain the distinction.
- pp.42–44, 50: classifier outputs are logits, despite prose calling them probabilities. Cross entropy combines log-softmax with negative log likelihood; it does not require an explicit softmax layer.
- pp.54: pinned memory explanation is oversimplified. Explain host allocation and transfer separately in Phase 05.
- pp.62–71: dropout placement is a source architecture choice, not a universal constraint; BatchNorm is not guaranteed to prevent overfitting, and adaptive optimizer decay needs a careful distinction from L2.
- pp.74–79: retain the Pima diabetes/random-forest case, invalid zeros, conditional SVM/RF/boosting spaces, samplers, plots, dataframe and ecosystem material. Fit preprocessing within CV folds to prevent leakage.
- pp.81–87: ANN Optuna optimizes test accuracy in source code. Replace this with validation accuracy and explicitly explain the repair.
- pp.98–101, 114: flattening removes explicit locality bias, not information itself. CNNs are not universally translation or rotation invariant. An ANN can use positional feature order; swapping feature columns without retraining changes its computation (p.143).
- pp.103–104: example feature-map diagram and written arithmetic conflict. Top-left sum is 0 with zero bias; verify the entire matrix in Phase 07.
- pp.119: source's feature-extractor parameter sum is inconsistent with its listed terms. Recalculate, including BatchNorm parameters.
- pp.123, 128–130: freezing features and retraining a new classifier is feature extraction; distinguish it from unfreezing backbone fine-tuning. `pretrained=True` is legacy; pp.140–141 already introduce the weights API and transforms.
- pp.150–151: Keras recurrent parameter counts use one recurrent bias; PyTorch exposes two. Preserve framework attribution.
- pp.165–166: ReLU's derivative is 1 on positive inputs; unstable recurrence depends on Jacobian products, not just individual weights.
- pp.170–179: the actual FAQ example predicts one answer **token** from vocabulary-sized logits. Phase 10's requested answer-class head is an explicit engineering improvement that supports complete multiword answers. Softmax confidence cannot guarantee rejection of unseen questions.
- pp.181–198: story slots are an analogy, not literal semantic memory cells. LSTM gates mitigate vanishing gradients; they do not guarantee preservation over arbitrary lengths. Exact 0/1 gates are idealized limits for finite sigmoid inputs.
- pp.200–211: source uses one ID for unknown and padding and downloads NLTK data silently. Separate special tokens and avoid hidden downloads. Top-k is already mentioned in source p.204; temperature sampling is supplemental. Source's 289-word, 942-sequence shape example refers to a larger corpus than its abbreviated inline code.

## Phase 07 source review

- pp.89–97: preserve the Fashion-MNIST CNN motivation, NCHW custom dataset, two Conv-ReLU-BatchNorm-Pool blocks, dense head, CrossEntropy training/evaluation flow and reported metrics. The CSV import and loader lines are clipped; reconstruct the tensor contract without guessing file URLs. Treat 92.62% test and 99.99% train as source-run figures.
- pp.98–101: flattening retains pixel values but removes explicit spatial axes from the model input. Convolution provides local connectivity and shared weights. The source claims translation invariance too strongly; convolution is conditionally equivariant and pooling adds only limited local robustness.
- pp.103–104: the shown output grid conflicts with the written top-left sum. Use an independent checked example and compare manual cross-correlation with torch.nn.functional.conv2d; clarify PyTorch does not flip the kernel.
- pp.105–108: verify floor behavior for padding/stride. The dilation-aware formula is a labeled supplemental generalization.
- pp.109–112: one filter spans all input channels; output channels equal filter count. BatchNorm2d affine parameters are 2C; running statistics are buffers. Preserve source block order Conv→ReLU→BatchNorm2d.
- pp.113–116: pooling has no trainable parameters and does not guarantee broad translation/rotation invariance. Global average pooling is an explicitly supplemental alternative to the source Flatten/dense head.
- pp.119–121: feature extractor total 19,008 includes Conv parameters 320+18,496 and BatchNorm affine parameters 64+128. The source custom trace method is replaced with normal model(x) and temporary hooks.

## Phase 08 source review

- pp.121–131: preserve source/target transfer, feature freezing, classifier replacement, training/evaluation modes, optimizer scope and reported compute limits. The source uses the term fine-tuning for a frozen feature extractor; identify this as feature extraction and define true fine-tuning separately.
- pp.124–139: reconstruct the eight-step CSV path. Keep PIL HWC uint8 RGB conversion separate from torchvision resize/crop/ToTensor/Normalize. `ToTensor` already scales uint8 bytes; do not pre-divide by 255. The exported CSV/loader lines are clipped and should not be copied verbatim.
- pp.137–141: verify channel means/std and the red example `(0.8−0.485)/0.229≈1.376`. Prefer the same `VGG16_Weights` enum for model and transform. Fashion-MNIST grayscale PIL images must be explicitly converted or replicated to RGB before the ImageNet preset.
- Source training reports about 94% from a prior run and a separate session that stopped at epoch six. Keep these observations contextual; use validation for checkpoint selection and test once as an explicit engineering repair.

## Phase 09 source review

- pp.142–147: preserve sequence motivation, ordered NLP/audio/time-series/DNA use cases, and the dense ANN limits of fixed widths, padding, truncation and one-hot inputs. Dense embeddings and chronological validation for time series are supplemental engineering context.
- pp.148–153: input is `[batch,time,features]` when configured with `batch_first=True`; `h_n` remains `[layers·directions,batch,hidden]`. Recurrent weights are shared across time. The source's I=5,H=3,O=1 count is 31 for its Keras single-bias SimpleRNN plus Dense head; PyTorch `nn.RNN` uses `bias_ih` and `bias_hh`, so equivalent count is 34.
- pp.153–158: keep one-to-one as a feed-forward contrast; explain many-to-one sentiment, one-to-many caption/music, aligned many-to-many tagging, and variable-length encoder-decoder translation. Question answering is a sequence application; BERT is a Transformer-era example, not evidence that a vanilla RNN is required.
- pp.158–162: preserve the toy sentiment BPTT sequence and shared-weight gradient accumulation. Reconstruct using `nn.Module`, `model(x)`, raw logits with `BCEWithLogitsLoss`, matching float targets, `zero_grad`, `backward`, optimizer step and evaluation contexts; synthetic snippets demonstrate mechanics, not sentiment quality.
- pp.163–167: reason about products of recurrent Jacobians. Tanh derivative is `1−tanh²(z)`, equal to one at zero and approaching zero in saturation. ReLU derivative is one on positive inputs. Matrix directions and norms matter; clipping limits exploding gradients but does not solve vanishing gradients. Gated cells and truncation have limits rather than memory guarantees.

## Phase 10 source review

- pp.168–169: retain the small general-knowledge FAQ demonstration, question tokenization, vocabulary construction and unknown-token fallback. The source reports a vocabulary size for its own data; do not present that dataset-specific number as universal.
- pp.170–173: the source `SimpleRNN` embeds IDs, runs `nn.RNN(batch_first=True)`, selects the final hidden state and maps to vocabulary-sized logits. Training uses CrossEntropyLoss and an answer target selected from answer token IDs; inference applies softmax/argmax and a maximum-score threshold. The result is one answer **token**, not a complete answer string, and maximum softmax is not reliable OOD detection.
- pp.174–179: preserve the Dataset/DataLoader, embedding/RNN/head, optimizer loop and prediction path. Exported DataFrame sample strings and model code are visibly clipped; reconstruct a complete local CPU example rather than guessing missing text or URLs. Source batch size one avoids padding; the batch collator with PAD, UNK and retained lengths is supplemental.
- Supplemental repair: make answer-class IDs map to known multiword responses, separating the input vocabulary from the response map. Fit vocabulary on training questions, save mappings with weights, and treat synthetic validation strictly as a pipeline demonstration. The interactive fixed FAQ mapping is hand-authored conceptual content, never a trained prediction.

## Phase 11 source review

- pp.180–187: preserve the motivation, story analogy, distinct hidden/cell roles, three gate purposes, concatenated inputs and vector-width relationships. The analogy is instructional; the cell is still modified at every step.
- pp.188–191: equations are `f=σ(W_f[h_prev,x]+b_f)`, `i=σ(...)`, `g=tanh(...)`, `c=f⊙c_prev+i⊙g`, `o=σ(...)`, `h=o⊙tanh(c)`. For finite logits, sigmoid values are strictly between zero and one; exact endpoints in the source's thought experiment are idealizations.
- pp.192–197: retain the three-coordinate arithmetic: forget contribution `[0,5,3]`, candidate contribution `[0.8,0,0]`, updated cell `[0.8,5,3]`; `tanh(c)≈[0.664,0.9999,0.995]` and output-gated hidden is approximately `[0.664,0,0.498]`. The PDF's rounded third tanh value `.99` is coarse; standard two-decimal rounding gives `1.00` for `tanh(5)`.
- p.198: the direct cell-state partial derivative with gate values held fixed contains the forget factor, and its multi-step direct route contains their product. This is not the full LSTM Jacobian: gates depend on recurrent quantities and other paths contribute. LSTMs can mitigate vanishing gradients; they do not guarantee arbitrary memory or eliminate exploding/vanishing behavior.
- Supplemental PyTorch context: native fused gate order is i,f,g,o; `nn.LSTM` state layout remains `[layers×directions,batch,hidden]`; default modules use two 4H bias vectors per direction/layer. Parameter counts, packed true lengths, chunk-state carry and synthetic classifier smoke tests are engineering additions, not quoted source lessons.

## Phase 12 source review and final manual audit

- pp.198–204: the source transitions from LSTM gates into next-word language modeling, demonstrates progressive prefixes, corpus-specific vocabulary/corpus counts, leading padding, an embedding–LSTM–linear stack, CrossEntropyLoss, Adam and autoregressive argmax. It suggests displaying top-three alternatives. Source values such as 78 sentences, 942 prefixes, 289 vocabulary entries, 50 epochs and 61 input steps describe that run only.
- pp.205–211: the exported complete script downloads NLTK tokenizer data, uses a tiny FAQ corpus, builds Counter IDs, generates prefixes, left-pads, splits final input/target columns, uses Dataset/DataLoader, trains a batch-first LSTM and appends predicted tokens. Several code lines are visibly clipped in the export; none are copied as valid code without reconstruction. The new CPU CLI is a complete local reconstruction.
- Source repair: the export assigns `<unk>` ID 0 and also uses leading zero padding, conflating an unseen word with an absent position. The project gives PAD, UNK and EOS distinct IDs. The NLTK download is replaced with a documented local regex tokenizer; this changes tokenization boundaries and is labeled as an implementation choice.
- Data-quality additions: sentence groups are split before prefixes are expanded, the vocabulary is fitted on training sentences only, validation loss is measured without updates, and configuration/vocabulary accompany the state dict. The source does not provide a held-out learning-curve protocol. The small corpus is a pipeline demonstration, not a language-quality benchmark.
- Inference: preserve source left-padding and explain that ordinary LSTM steps still process those PAD positions. Unknown inputs map to UNK; EOS and a maximum generation bound stop generation. Temperature-scaled categorical sampling is supplemental; top-k suggestions are source-derived. The illustrative distribution and append-stepper do not run Python.
- Full manual coverage audit: the page index contains ordered extracts for pp.1–211; `scripts/check-coverage.mjs` verifies that every page has at least one mapped topic. The final coverage manifest records 16 source blocks and reviewed lesson links for all mapped source topics. The release pass also resolved five earlier manifest entries that were still marked planned even though Phase 04 had authored coverage (Sequential/summary, samplers, collation, loader options and the broken dataset source/fallback). Nine grouped source-correction entries and all supplemental records are shown in the Source Coverage view.
- Artifact hygiene audit: indexed lesson/search content is checked for YouTube/Gemini terms, timecode-shaped strings and `&t=` links. Source-page extracts remain audit-only and are not bundled. The standalone build check rejects external asset URLs and runtime `fetch`; browser tests load the built `file://` artifact offline.

## Phase 01 preview scope

Three narrow previews show autograd's scalar derivative (pp.1–3), manual gradient descent (pp.8–9, 13–14), and sample-to-batch flow (pp.25–28, 39–40). Their toy data, reusable templates and assertions are editorial additions, marked in the reader; later lessons complete the corresponding source clusters. Original source conversation chrome, links and time references are excluded from bundled content.
