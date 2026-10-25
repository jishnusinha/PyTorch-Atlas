# PYTORCH ATLAS
## 12-Phase Cursor / Codex Implementation Prompt Pack

**Product:** PyTorch Atlas — an offline-first, deeply explained, visually rich, revision-friendly interactive technical textbook.

**Authoritative source:** `PYTORCH Guide - Google Gemini.pdf` (211 pages), placed in the **repository root** by the user. It is a PDF export of Gemini notes and code based on CampusX lessons, not a verified verbatim transcript of the original videos. The notes begin at an Autograd lesson and end at an LSTM next-word predictor. Do not claim access to videos or copy their timestamps.

**Final deliverable:** `dist/PyTorch-Atlas.html`, a self-contained, network-independent HTML file users can open directly (including via `file://`) to study, navigate, search, revise, view diagrams, and copy runnable Python boilerplate. Source code remains a maintainable React/TypeScript application. No server or runtime dependency is required to view the final HTML. The original reference PDF is not embedded in the distributed single HTML.

## How to operate this pack

1. Create a new repository, and put the original PDF in its root with exactly the filename above.
2. Paste **Phase 01** into Cursor/Codex Agent mode. Review the implemented files and run its acceptance tests.
3. Paste one subsequent phase at a time. Each phase refers to the source-audit records and architecture created in Phase 01. The agent must inspect existing code before editing; it must NOT throw away previously built sections.
4. After each phase: ask the agent to run tests and build the standalone HTML; review a screenshot or browser preview; commit only after verification. Do not push/deploy unless requested.
5. At any point, the agent must stop after completing that phase and report its actual changes; it must not silently begin the next phase.

## Shared implementation contract (place in `AGENTS.md` during Phase 01)

**Source fidelity**
- Read the relevant PDF pages *in full* before writing each lesson. Preserve concept coverage and order. Make a `docs/source-coverage.json` manifest for page spans and lesson-to-source mappings, including a supplemental/source-derived classification for each lesson. Cross-check every PDF topic, all important equations, training steps, and code patterns. Some exported code blocks have visually truncated lines: do not reproduce unverified incomplete code. Reconstruct examples and execute smoke tests.
- Source-derived means the concept is in the PDF; supplemental means important pedagogical or engineering context beyond it. Label supplemental material in the reader. Never present additions as if quoted from the PDF.
- Do NOT show YouTube/video timestamps, upload dates, timecodes, links with `&t=...`, Gemini chat UI, raw exported screenshot chrome, or conversation artifacts anywhere in the site, source snippets, section names, search index, metadata, print output, or generated descriptions. Page numbers **are** allowed, in the form `PDF pp. 25–29`.
- Preserve relevant use-cases in the PDF: Wisconsin breast cancer binary classification, Fashion-MNIST ANN/CNN, Optuna for tabular/ANN hyperparameter tuning, VGG16 transfer learning, simple FAQ classifier styled as question answering, and LSTM next-word prediction. Explain the toy/demo nature of the QA model.
- Treat source performance numbers as reported examples rather than guaranteed benchmark results. Distinguish PyTorch best practices and repairs from Gemini's original code.

**Educational content contract, EACH concept/lesson**
- Clear learning objective and why/when it matters; intuitive analogy; correct mathematical explanation (where relevant); a worked numerical example; explicit input/intermediate/output tensor shape traces; annotated SVG diagram with meaningful labels and legend; one concise runnable Python code example plus a reusable boilerplate template; line-by-line explanation; expected output or assertion; common failures and debugging tips; production notes where relevant; quick revision card; and 3–5 retrieval questions with revealable answers. Avoid thin placeholder cards and long uncategorized walls of text.
- PyTorch sample code should import needed modules, use `float32` unless justified, set seeds when randomness matters, use `model(x)` rather than calling `forward` directly, match target shapes and dtypes, use proper `train()`/`eval()` and gradient contexts, and avoid unintended train/test leakage. Use logits-compatible loss by default (`BCEWithLogitsLoss`, `CrossEntropyLoss`). When the PDF uses `Sigmoid+BCELoss`, show both with a careful comparison. Provide CPU-first runnable examples; optional GPU path must degrade cleanly.
- Avoid hidden downloads in first-run sample snippets. Generate small synthetic data when possible. Dataset-based labs should have well-marked optional downloads or local-file alternatives and reproducible setup.
- Every snippet has a **Copy** control, an optional **Download .py** control, dependency labels, a concise explanation of where to paste/run (Colab, local Python), and a disabled-by-design browser execution claim: Python examples are **not** run in the HTML page.

**Visual quality and accessibility**
- Design direction: editorial technical atlas, polished, minimal, extremely readable. Avoid huge cards, gratuitous gradients/glassmorphism, clutter, gimmicky animations, distracting badges, or generic dashboard aesthetics. Provide a sophisticated light theme and a carefully balanced dark theme; use system or bundled open fonts (no CDN required). Design tokens for surfaces, backgrounds, colors, borders, spacing, typography, code blocks, graphs, math, callouts, and focus states.
- Desktop: fixed-width/scrollable left navigation (about 270px), article content with comfortable max reading width (~780–900px), optional right-hand sticky in-section TOC (~210px) on wide screens. On smaller viewports use a keyboard-accessible drawer and single-column reading layout. Provide generous whitespace and numbered instructional steps.
- Site navigation: module groups, expandable lessons, active-section tracking, global text search with highlighted matches, title/path breadcrumbs, previous/next, keyboard shortcut `/` to search, progress checkboxes persisted locally, mark/unmark completion, bookmarks, resume reading, current page reading-progress indicator, and quick-reference/cheat-sheet view. Support reduced-motion, visible keyboard focus, semantic HTML, accessible diagram descriptions and minimum WCAG AA contrast where feasible.
- Diagrams: author vector SVG/React (no static blurry screenshots). Real content-driven diagrams, e.g. autograd DAG, tensor axes, backprop chain, data batching, ANN layers, optimization surface cross-section, convolution slider/stride, CNN dimension tracker, unrolled RNN, BPTT, gate-by-gate LSTM. Interactive controls should change the displayed conceptual state without attempting to run actual PyTorch in-browser. Include text alternatives.
- All book content lives in structured typed data (`src/content/...`, Markdown/MDX or typed TS content files) rather than one monolithic React component. Keep chapter IDs/slugs stable and let multiple phases extend an existing content manifest. Separate content, components, diagrams, code-snippet registry, and tests.

**Architecture and offline deliverable**
- Use Vite + React + TypeScript; configure a reliable one-file build such as `vite-plugin-singlefile` with locally bundled JS/CSS/icons and no runtime network calls. Avoid runtime dynamic `fetch()` of local JSON files; content must be statically bundled. Generate `dist/PyTorch-Atlas.html`. Test it under `file://` and with network disabled. If localStorage is unavailable under a particular browser policy, allow a graceful in-memory fallback. Clipboard has a permission-aware fallback for `file://`; show accurate success/failure messages.
- Suggested layout:
  ```
  /
  ├── PYTORCH Guide - Google Gemini.pdf
  ├── AGENTS.md
  ├── package.json / vite.config.ts / tsconfig.json
  ├── src/
  │   ├── app/                 # App shell, router/navigation, search and progress
  │   ├── content/             # Module manifests, lessons, revision items
  │   ├── components/          # Reader, sidebar, search, code, quiz, TOC
  │   ├── diagrams/            # Standalone SVG visualizations, accessible
  │   ├── design/              # Theme tokens and global CSS
  │   └── utils/               # Clipboard, download, local persistence
  ├── python_examples/        # Runnable mirrored Python code
  ├── tests/                  # UI, content, smoke tests
  ├── scripts/                # Check coverage, snippets, packaging
  ├── docs/source-coverage.json
  └── dist/PyTorch-Atlas.html
  ```
- Use a reproducible lockfile. Do not create a backend, require API keys, or provision cloud resources. Offline means no external images, scripts, CSS, fonts, Google Analytics, or telemetry. Avoid `eval()` and unsanitized raw HTML.
- Each phase must end with: changeset/file summary, PDF pages covered and supplemental items, sample snippets verified, number/names of working SVG diagrams, accessibility/responsiveness checklist, build/test commands with actual results, and unresolved issues. Preserve user changes. Do not push or deploy automatically.

---

# PHASE 01 — SOURCE AUDIT, DESIGN SYSTEM, CONTENT ENGINE AND STANDALONE BUILD

**Copy everything between START and END.**

### START PROMPT 01
You are acting as lead technical educator, PyTorch developer, accessibility-minded product designer, and principal frontend engineer. BUILD Phase 01 of a new project named **PyTorch Atlas**. The user has placed `PYTORCH Guide - Google Gemini.pdf` (211 pages) in the repository root. Inspect the repository first. Read or programmatically extract the complete PDF in page order. Derive a precise source index rather than guessing. Do not start writing complete later chapters yet.

1. Create `AGENTS.md` encoding this complete **self-contained baseline contract**: (i) PDF-first, read all relevant pages before each lesson, maintain topic/page/source-derived-or-supplemental traceability; (ii) never copy video timestamps, video links, Gemini conversation UI, or truncated code into the handbook; (iii) every concept must have learning goals, intuition, math/numerical walkthrough where relevant, shape trace, original accessible SVG diagram, annotated complete copyable runnable Python snippet with expected output and dependencies, real pitfalls, revision card and 3–5 answerable questions; (iv) elegant, accessible, responsive, distraction-free reading UI with left sidebar, search, progress, bookmarks, and light/dark mode; (v) offline single HTML deliverable with all JS/CSS/content inlined and no runtime server/CDN/fetch/tracking, file:// support and resilient clipboard/localStorage fallbacks; (vi) React+TypeScript+Vite typed content architecture with diagrams and Python examples separated into directories; (vii) `nn.Module`, `model(x)`, proper loss/logit conventions, train/eval and leak-free validation, CPU-safe executable checks; (viii) do not mutate the PDF, deploy, or push; (ix) write reproducible tests and a truthful report after every phase. Establish `docs/source-coverage.json` mapping every meaningful PDF topic to exact PDF page ranges, the future lesson slug, and status (`planned`, `draft`, `reviewed`), plus `source-derived`/`supplemental` flag. Split the document into source blocks: Autograd pp.1–5, manual training pp.6–14, nn.Module pp.15–24, Dataset/DataLoader pp.25–40, ANN pp.41–49, GPU pp.50–59, regularization pp.60–71, Optuna overview pp.72–79, Optuna ANN pp.80–88, CNN implementation pp.89–97, CNN visual/mathematical explanation pp.98–121, transfer learning pp.121–141, RNN theory/variants/BPTT/limitations pp.142–167, RNN FAQ example pp.168–179, LSTM concepts/gates pp.180–198, LSTM next-word predictor pp.198–211. Boundaries deliberately overlap where a topic starts midway through a page.
2. Scaffold Vite + React + TypeScript and configure a self-contained output file `dist/PyTorch-Atlas.html` with inlined styles/scripts, no remote assets and no runtime fetch. Use named components and a typed content model (`Module`, `Lesson`, `LessonSection`, `DiagramSpec`, `Snippet`, `RevisionQuestion`, `SourceReference`). Keep the PDF itself at root, never embed its pages into the final distributable.
3. Build a beautiful responsive reading UI: editorial brand/header, collapsible left module navigation, expandable chapter lessons, roomy article typography, desktop sticky TOC, proper breadcrumbs, light/dark mode, reading progress, finished checkmarks, bookmarks, session resume, previous/next, global search dialog, `Ctrl/Cmd+K` and `/` keyboard search shortcuts. Implement persisted progress and theme with safe fallbacks. Deliver three realistic *fully authored preview lessons* and clearly marked pending lessons in navigation; do not fake a complete handbook.
4. Develop reusable `CodePanel` (syntax highlighting, Copy, Download .py, purpose/dependencies/shape labels, clipboard fallback), `Equation`, `IntuitionPanel`, `ShapeTrace`, `SourcePage`, `RevisionCard`, `Quiz`, and accessible `DiagramFrame`. Build one genuine SVG concept diagram such as the computation graph as a style sample.
5. Add accessible keyboard navigation, focus states, mobile sidebar/drawer, print styles, reduce-motion support, helpful empty search state, loading-free instant nav, no video timestamps anywhere in the generated document. Make the UI polished: whitespace, restrained color, precise spacing, typography hierarchy and responsive diagrams rather than crowded cards. Use icons from a locally bundled icon package or hand-authored SVG.
6. Write tests for navigation, rendered content, keyboard search, progress persistence, copy/download, light/dark, empty result, offline rendering, and file:// compatibility. Add a link checker and a `source-coverage` checker. Include README with build and open-file steps.

Deliver actual code and `dist/PyTorch-Atlas.html`, not a design-only proposal. Run checks and report exact results. Stop at Phase 01; do not develop phases 02–12 yet.
### END PROMPT 01

---

# PHASE 02 — PYTORCH TENSORS, AUTOGRAD AND COMPUTATION GRAPHS

### START PROMPT 02
Inspect the current PyTorch Atlas repository and `AGENTS.md`, then implement Phase 02 **without rebuilding or replacing the shell**. Read the full PDF pages 1–5 first and use those pages as primary content. Prepend clearly flagged supplemental tensor prerequisites because the PDF begins with Autograd (it does not fully teach tensors from first principles).

Build complete lessons in this order: (a) why tensors exist; scalar/vector/matrix/N-D tensor, shape, axes, dtype, device, contiguous views and storage, broadcasting, tensor creation, indexing, `.reshape()`/`.view()`/`.unsqueeze()`/`.squeeze()`, `matmul` and common dtype/shape pitfalls (**supplemental**); (b) derivatives, gradient intuition and chain rule; (c) `requires_grad` and computation DAG, leaves/intermediates/loss root; (d) scalar autograd through `y=x**2` and `z=sin(x**2)` with manual derivatives verified against `backward()`; (e) vector reduction and multi-input gradients, `.grad`, gradient accumulation, `.retain_grad()` when useful; (f) `no_grad()`, `inference_mode()`, `.detach()` and how they differ (**supplemental** where not directly in source).

Teach tensor shapes for every code block and demonstrate a gradient graph step by step. Required diagrams: tensor axis/shape visualizer, computation graph with forward values and backward gradients, chain-rule flow, graph tracking vs disabled. Interactive controls: scalar x slider changing f(x) and f'(x); an on/off graph-tracking toggle that changes conceptual depiction (not real PyTorch execution). Make every diagram accessible with fallback explanations and meaningful labels.

Author at least 12 complete standalone CPU-first Python snippets, including tensor shape transformations, broadcasting, numeric derivative comparison, gradient accumulation + clearing, non-scalar backward with `sum()` or gradient argument, `detach` and inference contexts. Mirror runnable snippets in `python_examples/02_foundations_autograd/` and smoke-test assertions with pytest. Ensure no code requires a dataset download. Code should be explained line by line with expected output. Provide a one-screen Autograd cheat sheet, common error explanations (non-scalar backward, in-place ops on leaves, missing `.grad`, mixed types) and knowledge-check questions.

Update content index and source coverage statuses, preserve progress/search and previously implemented preview content. Build and test `dist/PyTorch-Atlas.html` with zero video timestamps. Report results and stop after Phase 02.
### END PROMPT 02

---

# PHASE 03 — COMPLETE MANUAL TRAINING PIPELINE, FROM LOSS TO PARAMETER UPDATE

### START PROMPT 03
Inspect the current application and existing lessons. Read PDF pages 6–14 in full, including the source breast cancer binary classification notebook and manual code; verify visually where the PDF export cuts off the long lines. Implement Phase 03 in the live handbook and mirrored testable Python examples. Do not merely paste the source code: reconstruct working code and explain each operation.

Create ordered lessons: problem framing and data labels; loading and identifying features/target; train/test separation and scaling with `fit` on training only; tensor conversion and dtype; manual `MySimpleNN` storing trainable `weights` of shape `[30,1]` and `bias`; forward `X @ W + b`, sigmoid and `[batch,1]` prediction shapes; binary cross-entropy from first principles with clipping and limitations; chain-rule path from loss to weights; `loss.backward()`; manual SGD in `torch.no_grad()` followed by zeroing accumulated gradients; 0.5 classification threshold; evaluation and confusion-table interpretation. The PDF uses a Breast Cancer dataset and manual BCE; show that approach faithfully and explain how `BCEWithLogitsLoss` later improves numerical stability. Add **supplemental** validation split, `float32` practice, label-direction verification, stratification and data leakage prevention.

Build a central interactive *training pipeline explorer*: numbered stages from raw row -> scaled features -> tensor -> logits -> probability -> loss -> backward gradient -> updated parameters -> prediction; clicking a stage displays shapes, equation, relevant snippet and expected values. Add a graph showing the relationship between loss/learning rate/update direction; teach why `zero_grad` matters with two mini iterations.

Supply runnable standalone snippets for preprocessing (with a synthetic-data fallback), manual forward, manual BCE, one manual train step, full clean CPU binary classifier and evaluation. Keep an optional local CSV input path instead of a fragile hardcoded external URL. Python smoke tests must assert decreasing loss on toy linearly separable data and shape/dtype correctness; optionally support an explicit Breast Cancer dataset download command documented separately. Do not pretend downloaded data is bundled if it is not.

Content per lesson: detailed intuition, math, real numbers, tensor shapes, annotated code, mistakes, revision cards, 3–5 questions. Update sidebar, manifest, search and build. Stop after completing Phase 03 and reporting tests.
### END PROMPT 03

---

# PHASE 04 — nn.Module, nn.Sequential, DATASET AND DATALOADER

### START PROMPT 04
Read the present repo and `AGENTS.md`; implement Phase 04 against PDF pages 15–40, which evolve the manual training code into idiomatic PyTorch and add mini-batching. Do not delete the manual implementation from Phase 03: make an explicit manual-vs-idiomatic comparison page.

Develop full lessons for `nn.Module` inheritance; why `super().__init__()` is required; register and inspect `nn.Linear` and `nn.Parameter`; `forward` vs calling `model(x)` and module hooks; `model.parameters()` / `state_dict`; `nn.Sequential` vs flexible explicit `forward`; activations; `BCELoss` vs recommended `BCEWithLogitsLoss` with logits; class/target shape errors; PyTorch `torch.optim.SGD`, `zero_grad(set_to_none=True)`, `backward`, `step`; `torchinfo`-style architecture summary with optional dependency. Then Dataset & DataLoader: `__init__`, `__len__`, `__getitem__`, `TensorDataset` vs custom `Dataset`, `batch_size`, shuffle, `RandomSampler`/`SequentialSampler`, custom `collate_fn`, padding, `drop_last`, workers, pin memory (conditional), train vs eval loading, iterable datasets (**supplemental** if needed), Windows multiprocessing cautions (**supplemental**).

Show shapes across a 30-input-feature binary model and batch pipeline. Interactive diagrams: module parameter-registration tree, dataset index lookup -> sampler -> batch -> collate -> model, and gradient update lifecycle. Include a batch-size slider (e.g. 10 records, sizes 1–5) showing how many batches and what `drop_last` changes. Add readable code side-by-side comparisons: manual weight-update vs SGD; manual class vs `nn.Module`; Dataset vs DataLoader.

Produce at least 12 runnable boilerplates, including stable binary classifier, Dataset, TensorDataset, DataLoader, dynamic text `collate_fn`, `state_dict()` save/load with safe file IO and examples of correct train/eval modes. Provide unit tests and short CPU smoke run using synthetic arrays. Show real errors and resolutions (mismatched target `[B]` and logits `[B,1]`, `num_workers` behavior, string labels, incorrect dtype). Do not leave copied broken source URLs.

Update coverage, build standalone HTML, test key interactions, and stop after Phase 04.
### END PROMPT 04

---

# PHASE 05 — ANN IMAGE CLASSIFICATION, FASHION-MNIST AND GPU TRAINING

### START PROMPT 05
Inspect application, earlier content, and `AGENTS.md`; read PDF pp.41–59. Implement the ANN and GPU training chapters in the existing atlas. Explain the actual source workflow and clearly mark any newly introduced best practices.

Lessons: Fashion-MNIST rows and labels; image-to-vector flattening 28×28 -> 784; multilayer perceptron topology, nonlinear activations and logits for 10 classes; `CrossEntropyLoss` expecting unnormalized `[B,10]` logits and `long` targets `[B]`, not softmax input; 100-epoch/batch/learning-rate setup as a historical example, not a hard requirement; epoch + mini-batch training and evaluation; training curves, overfitting clues, accuracy and confusion matrix; reliable device selection (`cuda`, `mps`, CPU where supported), parameters/data to same device, `.to(device)`, CPU/GPU transfer overhead and correct interpretation of `pin_memory=True` vs `non_blocking`; GPU memory use and safe fallbacks. Add supplemental validation and checkpoint flow, reproducible seeding, performance profiling basics, balanced accuracy/F1 cautions.

Must include ANN layer-graph and parameter counter, tensor path `[B,1,28,28] -> [B,784] -> [B,H1] -> [B,H2] -> [B,10]`, train/validation/test flow, CPU/GPU device-placement diagram, confusion-matrix visual, training-vs-validation curve. Interactive feature: user changes hidden widths and sees parameter count update; all computed counts correct.

Provide runnable Python code: synthetic-data standalone ANN smoke test with no downloads, and **separately** a Fashion-MNIST `torchvision` implementation with an explicit optional dataset-download step, training/eval loops, cross entropy, and device handling. Include helper boilerplates for metric aggregation, checkpoint saving, and shape assertions. Avoid full 100-epoch training as a test. Test tiny batches CPU-only. The HTML is conceptual and revision-oriented, not a Python runner.

Ensure accessible visuals, source pages for each topic, revision questions, completed coverage, stable one-file build and no external runtime calls. Stop after Phase 05.
### END PROMPT 05

---

# PHASE 06 — GENERALIZATION, REGULARIZATION, OPTIMIZERS AND OPTUNA

### START PROMPT 06
Inspect repo and read PDF pages 60–88, including the follow-up deep explanations, generic Optuna tutorial and ANN hyperparameter tuning. Integrate Phase 06 in order. The PDF covers dropout, batch normalization, weight decay, Bayesian/TPE/other search ideas, Optuna studies/trials/objectives/samplers/plots and the eight ANN hyperparameters. Preserve these distinctions; label advanced engineering additions as supplemental.

Lessons: underfit vs overfit and reading train/validation curves; dropout as stochastic masking during train vs no dropout in eval, rescaling intuition; BatchNorm1d mean/variance, epsilon, learnable gamma/beta, running stats, train/eval behavior and small-batch limitations; L2 regularization and `weight_decay` with caveat that decoupled weight decay in AdamW is not identical to L2 in adaptive optimizers; SGD, momentum, Adam and AdamW differences (**expanded**); optimizer zero/backward/step order; hyperparameters vs learned weights; train/validation/test protocol and why the test set must NOT be optimized against; grid/random/TPE search; Optuna `trial`, objective, search spaces (`suggest_int`, `suggest_float(log=True)`, `suggest_categorical`), sampler, pruner, study, storage, reproducibility, callbacks, plots and parameter importance. Recreate the ANN search over hidden layers, units, epochs, optimizer, LR, batch size, dropout, weight decay while balancing search cost. Make clear that TPE is a model-based sampler and avoid the simplistic claim it mathematically guarantees the global optimum.

Visualizations: interactive overfitting curves, dropout neuron-mask explorer, batch normalization gamma/beta slider with actual numeric before/after values, weight-decay vectors, optimizer movement on a 2D loss landscape with clearly labeled *illustrative* paths, hyperparameter search-space map, Optuna trial comparison with parallel coordinate view. Use lightweight local SVG/canvas, not remote plotting APIs. Add experiments log/revision table and readable cheat sheets for loss/optimizer/scheduler.

Runnable code: dropout and BatchNorm toy shape demonstration, comparison of regularization, minimal reproducible Optuna synthetic objective, Optuna objective training only on training split, validating on validation split, best-params refit, untouched final test, safe optional plotting commands. Tests must assert correct train/eval behavior and catch test leakage. MLflow integration may be shown as **optional supplemental** rather than a required dependency; never require server/API key. Keep all content offline in final HTML. End with source coverage and actual tests, then stop.
### END PROMPT 06

---

# PHASE 07 — CNN FROM FIRST PRINCIPLES THROUGH COMPLETE CLASSIFIER

### START PROMPT 07
Read `AGENTS.md` and PDF pp.89–121 completely, including the unusually detailed follow-up explanation (pp.98–121). Build Phase 07 as a first-class visual CNN learning lab. Preserve the original Fashion-MNIST CNN use case and compare it with ANN; do not skip the deeper convolution math.

Lessons in order: why flattening destroys spatial locality; images as `[N,C,H,W]` tensors, grayscale vs RGB; kernel/filter intuition, cross-correlation vs mathematical convolution, multiply-and-accumulate; parameters and shared kernel weights; padding, stride, dilation (**supplemental**), spatial output formula `floor((n + 2p - d(k-1) - 1)/s + 1)` and no-dilation simplification; channels and filters, `[out_ch,in_ch,kH,kW]` weights and parameter formula; feature map size calculations; Conv2d, ReLU, BatchNorm2d, MaxPool2d, AvgPool2d, adaptive/global pooling; receptive fields, flatten vs global-average-pool heads; full two-block Fashion-MNIST CNN and model.summary/parameter check; CNN vs ANN generalization and efficiency. Recreate the PDF's layer-by-layer shape trace with numerical examples and a dimension table.

Diagrams MUST include: 28×28 pixel grid + moving 3×3 sliding kernel with numerically correct elementwise products and sums; padding/stride toggles; multi-channel filter volume with channels; 2×2 pooling example; receptive-field expansion; CNN layer flow with live output shapes, trainable parameter count, and a color-coded final classifier. Diagrams must be responsive, accessible and suitable for mobile; interactive changes must update actual math.

Provide at least 12 runnable standalone/boilerplate snippets: manual small convolution calculation (NumPy or pure PyTorch), `F.conv2d`, output-size function with assertions, multi-channel Conv2d, pooling, CNN `nn.Module`, dataset transform, forward shape logging, parameter counting, smoke-train on synthetic images, optional full Fashion-MNIST training pipeline with explicit download, and evaluation. Add shape mismatch debugger and cheat sheet. Smoke tests must validate diagrams' formulas against PyTorch tensor outputs for representative kernel, stride and padding cases.

Update sidebar/coverage/search; build self-contained HTML; test and stop after Phase 07.
### END PROMPT 07

---

# PHASE 08 — TRANSFER LEARNING, VGG16 AND IMAGE PREPROCESSING

### START PROMPT 08
Read source PDF pp.121–141 and inspect existing CNN module before implementing Phase 08. Create a detailed visual tutorial on transferring a pretrained VGG16 classifier to Fashion-MNIST while clearly distinguishing pretrained feature extraction, classifier replacement, frozen feature extraction, and later optional fine-tuning. Validate architecture-specific details against the version of installed torchvision rather than assuming outdated APIs.

Explain why original Fashion-MNIST 28×28 grayscale/flat CSV cannot directly enter VGG16 (channel count, normalization, spatial dimensions). Walk every data transformation with before/after exact shape/dtype/range: flatten -> 28×28 2D -> uint8 PIL grayscale -> RGB -> resize -> center crop -> tensor `[3,224,224]` -> normalize using correct pretrained weights metadata. Note where PDF uses a manual eight-step pipeline; preserve and explain it, but present `weights.transforms()` as a safer maintained option and discuss differences when augmentation is used. Explain pretrained `weights` enum, download/cache requirements, offline model weights caveat, `model.features`/`model.classifier`, `requires_grad_(False)`, optimizer parameter groups, and why class count becomes 10.

Build interactive side-by-side source-image and model-input shape funnels, before/after preprocessing gallery using generated schematic pixel blocks (not external photos), freeze/unfreeze parameter tree, VGG16 shape path and training workflow. Require a table of parameters trainable before/after freezing and conceptual compute/cost discussion. Explain why `.eval()` and `.train()` affect BatchNorm/Dropout even when weights are frozen, and how to avoid accidentally updating frozen feature statistics. Include clearly labeled supplemental alternatives (ResNet18, feature-extraction vs fine-tuning, augmentation correctness) without replacing source's VGG16 lesson.

Boilerplates: pure synthetic transform demonstration that avoids downloading weights; image conversion and VGG preprocessing; load VGG16 using modern `weights=` API; freeze features; replace classifier; optimize only intended parameters; safe evaluation; save best weights; CPU-friendly tiny tests. Full training is optional and must explicitly state ImageNet weights download requirement; never execute costly full VGG16 training during tests. All diagrams SVG/offline, all snippets copyable/downloadable and verified. Build/test and stop after Phase 08.
### END PROMPT 08

---

# PHASE 09 — RNN MOTIVATION, UNROLLING, TYPES, BPTT AND GRADIENT PATHOLOGIES

### START PROMPT 09
Read PDF pp.142–167 in full. Implement a comprehensive RNN theory section before any RNN project. The source discusses why MLP/CNN architectures struggle with variable-length sequences, recurrent hidden state, architecture variants, forward propagation through time, backpropagation through time (BPTT), and vanishing/exploding gradients. Preserve the sequence and teach mathematics carefully.

Lessons: sequence vs fixed-length input; tokens and embeddings (additional needed prerequisites clearly labeled supplemental); recurrent cell `h_t = tanh(W_xh x_t + W_hh h_(t-1) + b)`; hidden states, shared weights across steps, sequence lengths and batch-first conventions; many-to-one, one-to-many, many-to-many aligned and encoder/decoder; output at each time step vs final state; BPTT with an unrolled three-step graph and derivative chain across timesteps, gradient contribution from multiple losses, shared-weight gradients; why repeated Jacobian products cause vanish/explosion; gradient norm, clipping, truncation (**supplemental**), and why LSTM helps (preview only). Keep RNN use-case claims proportionate: simple RNNs are educational baselines, not today's default language-model architecture.

Must build rich SVG unrolling animation/stepper with token inputs, states and outputs; arrows showing forward vs backward flow; shape inspector for `nn.RNN(batch_first=True)` and `output`/`h_n` including multi-layer and bidirectional edge cases (**supplemental**); architecture variants interactive selector; gradient propagation simulator for scalar recurrent multiplier less than, equal to, greater than 1. All examples must be mathematically valid and visually label time steps clearly, with text alternatives.

Provide minimum 10 runnable CPU-safe PyTorch snippets: hand-coded recurrent update for 3 steps, equivalent `nn.RNN`, many-to-one classifier, all-time-step output, shape assertions, toy BPTT `backward()`, gradient clipping, padded sequences primer, and sequence training toy example. Compare outputs with numeric test expectations where possible. Deep explanation per lesson, debugging notes and retrieval cards. Update manifest and standalone build, run tests and stop after Phase 09.
### END PROMPT 09

---

# PHASE 10 — RNN QUESTION ANSWERING / FAQ INTENT CLASSIFICATION LAB

### START PROMPT 10
Read PDF pp.168–179 and the working code fully before coding Phase 10. Implement the educational 'question answering' RNN project as accurately represented: a compact FAQ question-to-answer-class classifier; explain explicitly that it selects among known answer classes and is **not** open-ended generative question answering, RAG, or a factual retrieval engine. Retain the source's project storyline and main PyTorch methods while improving robustness.

Teach step-by-step: sample FAQ pairs, text normalization, tokenization, vocabulary and index mapping, why out-of-vocabulary token ID is essential; `Dataset`/`DataLoader` items and target IDs; `nn.Embedding(vocab_size, embed_dim)`, `nn.RNN(embed_dim, hidden_size, batch_first=True)`, `nn.Linear(hidden_size, n_answers)`, `CrossEntropyLoss`, training labels as `long`, correct `[B,T] -> [B,T,E] -> [B,T,H] -> [B,H] -> [B,K]` tensor flow; final hidden-state selection vs padding-aware sequence lengths, optional `pack_padded_sequence` (**supplemental**), `argmax`/softmax with confidence interpretation, save/load vocabulary/model and inference. Explain why the source uses batch size 1 for different sequence lengths, then implement a proper batch collator as an improvement.

Diagrams: sentence-to-token stream, vocabulary map, embedded token matrix, unrolled RNN classifier, final-logit probability table, and end-to-end inference pipeline. Provide an interactive miniature FAQ demo in plain JavaScript using fixed *illustrative* outputs, marked clearly as conceptual, not a live PyTorch model. The interactive example must NEVER fabricate learned model predictions.

Code deliverables: complete CPU-runnable local toy dataset version (no external URL), vocabulary builder with `<pad>` and `<unk>`, model + `Dataset` + collate + loop + inference, quick unit tests on tiny examples, reproducible seed and checkpoint with mapping. Include overfitting/limited-data/memorization limitations. Source-page tags and supplemental markers, source coverage, review questions, test/build results. Stop after Phase 10.
### END PROMPT 10

---

# PHASE 11 — LSTM INTERNALS, GATES, CELL STATE AND PYTORCH SHAPES

### START PROMPT 11
Read PDF pp.180–198 carefully. Build Phase 11 as the most intuitive and mathematically accurate LSTM chapter of the Atlas. Preserve the PDF's storytelling analogy (short-term vs long-term plot elements and when to forget) but distinguish an analogy from literal neural-network memory operations. Follow with correct PyTorch implementation and shapes.

Teach: RNN long-range context limitation; hidden state `h_t` vs cell state `c_t`; gate roles, sigmoid/tanh ranges and elementwise multiplication/addition; forget `f_t = sigmoid(W_f[x_t;h_(t-1)] + b_f)`, input `i_t`, candidate `g_t`, update `c_t=f_t ⊙ c_(t-1)+i_t ⊙ g_t`, output `o_t`, and `h_t=o_t ⊙ tanh(c_t)`; how forget/input gates create a path for information and gradients; subtle caveat that LSTM mitigates but does not mathematically eliminate vanishing gradients. Explain vector dimensions and each gate's learned matrices. Cover PyTorch `nn.LSTM` input/output/(h_n,c_n) shapes for `batch_first`, multiple layers, direction count, `num_layers`, and bidirectional models. Clarify that h_n/c_n follow `(layers*directions,batch,hidden)` rather than `(batch,seq,hidden)`.

Build a centerpiece high-quality SVG LSTM cell, with a moving token through forget -> input -> candidate -> cell update -> output gates; sliders for forget/input/output values and a single cell dimension with numerically correct computations; separate RNN vs LSTM gradient route view; state-shape inspector; gate parameter count calculator with explicit note about PyTorch's two bias vectors per gate in each layer. Provide manual single-timestep implementation, explicit `nn.LSTM` example, tensor shape assertions, gradient check, mini sequence-classification demo and small codebook. Include 10+ self-contained Python snippets, math walk-throughs with hand-computed numbers, accessible text alternatives, quiz/revision cards.

Avoid embedding exported screenshots; diagrams are original editable SVG components. Complete source index and standalone build, automated tests, mobile/readability review; stop after Phase 11.
### END PROMPT 11

---

# PHASE 12 — COMPLETE LSTM NEXT-WORD PROJECT, FINAL COVERAGE AND RELEASE

### START PROMPT 12
Inspect the existing Atlas and read the PDF pp.198–211 in full (and pp.180–198 for context). Implement the final substantial project **LSTM Next-Word Predictor** and then perform a systematic full-manual release pass, covering all 211 PDF pages and all supplemental topics in the index.

Project lessons: corpus and limitations of tiny training data; tokenize and normalize; `Counter` and vocabulary/`<unk>`/`<pad>`; deterministic encode/decode; sliding prefix -> next-token examples; causal next-token target alignment and no target leakage; sequence padding/truncation and attention to how source left-pads the sequences; `Dataset`/`DataLoader`; `nn.Embedding`, `nn.LSTM(batch_first=True)`, `nn.Linear` vocabulary logits; full output and final hidden/cell state shapes (use an explicit worked reference such as `[32,61] -> [32,61,100] -> [32,61,150] -> [32,289]` as a source-based example, but keep dimensions configurable); CrossEntropyLoss; Adam optimizer, gradient clipping, train/validation learning curves; seed text to token IDs, inference mode, model.eval, next token argmax, top-k and temperature sampling (**supplemental**), iterative autoregressive generation, EOS/stop conditions and unknown-word behavior. Explain why language modeling with LSTM differs from a transformer-based LLM and do not claim the PDF teaches transformers.

Implement a complete local CPU-runnable project in `python_examples/12_lstm_next_word/` with a tiny inline corpus, CLI `train` and `predict` entry points, vocabulary saved along with state dict and configuration, reproducible seeds, dependency notes, small smoke tests; separate optional larger-corpus path without unconsented downloads. Show runnable copyable boilerplates in Atlas for each stage and a single consolidated script/project layout. Add a visual N-gram sequence creator, padding viewer, LSTM shape trace, top-k next-word distribution explorer with *illustrative* probabilities and an autoregressive token append stepper clearly not running Python.

**Final quality audit**: (1) inspect all PDF pages 1–211 and check every significant subject has lesson coverage; (2) generate an accessible `Source Coverage` view listing page ranges, exact covered lessons, supplemental additions and issues corrected; (3) source fidelity and mathematical test pass; (4) smoke execute all runnable Python snippets or mark explicit optional dependencies/data; (5) verify no truncated code pasted as valid code; (6) verify no video timecodes/timestamps or Gemini UI artifacts anywhere visible or in generated search index; (7) test 360px/mobile, tablet, desktop layouts, keyboard-only navigation, dark/light, print, diagrams and reduced motion; (8) verify all copy controls and .py downloads including file:// clipboard fallback; (9) build `dist/PyTorch-Atlas.html` and open locally with network disabled; (10) verify no external fetch/CDN/telemetry; (11) create `docs/verification-report.md` reporting tests, lesson count, diagram count, snippet count, known limitations and source gaps; (12) update README with commands and distribution instructions.

Deliver the fully authored revision handbook, not placeholders, and the final standalone HTML. Do not alter the original PDF. Do not push or deploy unless asked. Report concrete test and build outcomes and stop.
### END PROMPT 12

---

## Recommended content inventory (must be verified against actual PDF)

| Source-derived cluster | PDF pages | Reader's conceptual coverage |
|---|---|---|
| Autograd | 1–5 | gradients, graph, chain rule, requires_grad, accumulated gradients, no_grad/detach |
| Manual binary classification | 6–14 | preprocessing, 30-feature weights, sigmoid, manual BCE, gradient descent, evaluation |
| Idiomatic PyTorch | 15–24 | nn.Module, super, layers, Sequential, optimizers, code |
| Data pipeline | 25–40 | Dataset, DataLoader, batching, sampling, collate, worker, full code and fixes |
| ANN Fashion-MNIST | 41–49 | MLP multiclass logits, CrossEntropyLoss, train/test |
| GPU | 50–59 | device, to(device), pin_memory, full loop |
| Regularization | 60–71 | Dropout, BatchNorm, weight decay, intuitive explanations |
| Generic Optuna | 72–79 | sampler, trial, study, hyperparameter search, visualizations |
| ANN Optuna | 80–88 | architecture + optimizer tuning, eight hyperparameters, MLflow mention |
| CNN build | 89–97 | Fashion-MNIST conv/pool architecture, training and comparison |
| CNN deep dive | 98–121 | image tensors, convolution math, channels, shapes, padding, pooling, receptive field |
| Transfer learning | 121–141 | VGG16, frozen feature extractor, preprocessing pipeline, weights.transforms |
| RNN concepts | 142–167 | motivation, forward, variants, BPTT, vanishing/exploding gradients |
| RNN demo | 168–179 | FAQ classifier, vocabulary, embedding, train and predict |
| LSTM | 180–198 | long/short memory, gates, math, flow |
| LSTM next word | 198–211 | prefix sequences, padding, vocab, LSTM, train, autoregressive generation |

## Release definition of done

- `dist/PyTorch-Atlas.html` opens as a single local file with no network.
- Every lesson visible in the navigation has real authored educational content; pending lessons only during intermediate phases.
- Every PDF topic has a page-grounded mapping and each supplement is visibly marked.
- No video timestamps anywhere in generated UI, text, code comments, search content, or print view.
- All code copied from the site is syntactically complete and either runnable without downloads or explicitly tagged as requiring optional dependencies/data/weights.
- All diagrams are readable at 360px viewport width and have text alternatives.
- Search, progress, bookmarks, theme, sidebar, quiz reveal, code copy, and .py download work offline.
- Unit/content/E2E/Python smoke tests are run and documented, and known failures clearly reported.
