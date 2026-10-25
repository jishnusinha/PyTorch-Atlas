# Phase 01 verification report

Verified on 25 October 2026. This section records the original Phase 01 closeout; the Phase 02 closeout follows below.

## Delivered

`dist/PyTorch-Atlas.html` is a self-contained reader, approximately **331 KiB** (338.92 kB as reported by Vite). It was opened directly using a `file://` URL in Chromium with the browser context set offline. The distribution has all JS, CSS, structured content, syntax highlighting, and vector diagrams inlined. It requires no runtime server, remote assets, local JSON fetch, account, or API key. The source PDF is not embedded.

The source remains a formatted React + TypeScript + Vite project. `AGENTS.md` records the shared educational, source-fidelity, offline, accessibility, testing and phase-boundary contract. The directory is initialized as a local Git repository; no commit, push or deployment was performed.

### Changeset / file summary

| Area                | Files / result                                                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build               | `package.json`, `package-lock.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `scripts/package.mjs`                                                           |
| Source audit        | `docs/source-coverage.json`, `docs/source-audit.md`, `docs/pdf-page-index.json`, 211 full extracts in `docs/source-pages/`                                            |
| Content engine      | Typed `Module`, `Lesson`, `LessonSection`, `DiagramSpec`, `Snippet`, `RevisionQuestion`, and `SourceReference` in `src/content/`; separate searchable lesson registry |
| Reader              | `src/app/App.tsx`: navigation, breadcrumbs, section tracking, progress, completion, bookmarks, resume, previous/next, reference and coverage views                    |
| Learning components | `CodePanel`, `Equation`, `IntuitionPanel`, `ShapeTrace`, `SourcePage`, `RevisionCard`, `Quiz`, `DiagramFrame`; native modal search and highlighted results            |
| Visual system       | `src/design/`: system fonts, light/dark tokens, responsive layouts, visible focus, print and reduced-motion support                                                   |
| Diagrams            | `src/diagrams/`: original SVG layouts and conceptual controls; mobile-specific graph and batching compositions                                                        |
| Utilities           | Safe persistence, clipboard fallback, Python downloads in `src/utils/`                                                                                                |
| Python              | Six standalone scripts in `python_examples/01_previews/`, requirements, `scripts/smoke-python.py`                                                                     |
| Verification        | Content tests, 16 Playwright tests, coverage checker, dependency audit, screenshot and source/print review scripts                                                    |
| Instructions        | `README.md`, this report, `.gitignore`, formatting configuration                                                                                                      |

## Source coverage and authored previews

All **211 pages** were extracted in page order and read for the initial audit. The export's duplicated text layer was removed only in the reading helper, not in the preserved extracts. The manifest maps **122 topics across 16 source blocks**, plus **4 supplemental records**. This is an audited roadmap, not a claim that the later lessons are authored.

| Preview              | Source basis            | Supplemental additions                                                                                                                       |
| -------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Autograd intuition   | PDF pp. 1–3             | Fresh scalar example, derivative assertion, reusable derivative checker, local numerical walkthrough and diagram controls                    |
| One training step    | PDF pp. 8–9 and 13–14   | Scalar squared-error toy example in place of the source's full binary classification lab, learning-rate experiment, reusable scalar fit loop |
| Mini-batch intuition | PDF pp. 25–28 and 39–40 | Deterministic TensorDataset example, batch accounting, portable loader factory, explicit limitations of the count formulas                   |

Each preview includes objectives, motivation, analogy, mathematical/numerical walkthrough, tensor shapes, original accessible diagram, complete runnable example and reusable template, line explanations, expected output/assertions, pitfalls, practical notes, revision card and four revealable retrieval questions. Source references and supplemental additions are visible in the reader. All larger lessons remain Planned. Source-derived preview topics remain Draft at the full source-topic level because their broader chapters are still pending.

The audit records clipped code, the source's broken dataset URL, logit/probability terminology, validation leakage in tuning, CNN arithmetic/parameter discrepancies, Keras/PyTorch recurrent bias differences, the source FAQ model's actual single-answer-token head, and LSTM gradient caveats. These are documented for their corresponding future phases rather than silently presented as corrected complete chapters.

Source PDF SHA-256, recorded after read-only auditing:

`A07FA1D76EAC067829B4CBC6F0522C29395673107F1F96A34D7B427B216DF100`

## Working diagrams

**Three conceptual diagrams** are implemented, with two additional mobile SVG compositions:

1. **Scalar computation graph:** changing x updates x² and 2x, with labeled forward and backward paths. A vertical composition is used on mobile.
2. **Manual gradient descent:** changing the learning rate updates w = 3 − 4η, the plotted parameter and recomputed squared-error loss.
3. **Sample-to-batch grouping:** changing batch size and drop_last updates membership, kept/dropped counts and final-batch arithmetic. A two-row mobile composition keeps sample labels readable.

All diagrams have accessible names/descriptions and text explanations. They demonstrate their stated equations in JavaScript; they do not claim to run PyTorch in the page.

## Actual verification results

| Command / check                                   | Result                                                                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm.cmd run build`                               | **PASS** — strict TypeScript check and Vite standalone packaging; 338.92 kB HTML, 103.96 kB gzip reported by Vite                                                         |
| `npm.cmd test`                                    | **6 / 6 PASS** — lesson completeness, navigation IDs, mirrored snippets, substantive search text, excluded source chrome/time references, standalone asset/network checks |
| `npm.cmd run check:coverage`                      | **PASS** — all 211 pages have extract and topic mappings; 16 blocks, 122 topics, 4 supplements                                                                            |
| `npm.cmd run test:e2e`                            | **16 / 16 PASS**, 33.9 seconds — built HTML opened using file URLs with networking disabled                                                                               |
| `./tmp/python/python.exe scripts/smoke-python.py` | **6 / 6 PASS** — Python 3.12.10, PyTorch 2.14.1+cpu; mathematical and shape assertions executed                                                                           |
| `npm.cmd audit --cache .npm-cache`                | **0 vulnerabilities** — the initially reported PDF.js development-tool advisory was fixed by updating to 6.2.108                                                          |
| `node scripts/render-source.mjs`                  | **PASS** — source pages 3, 8, 13, 28 and 104 rendered; final print samples rendered for visual QA                                                                         |
| `node scripts/review-reader.mjs`                  | **PASS** — offline desktop overviews and mobile graph/update captures                                                                                                     |

The six verified Python files and outputs:

| Script                | Observed output                     |
| --------------------- | ----------------------------------- |
| `scalar-gradient.py`  | `value=9.0, gradient=6.0`           |
| `derivative-check.py` | `6.0`                               |
| `manual-step.py`      | `weight=2.60, loss=2.56`            |
| `manual-loop.py`      | `weight=1.023`                      |
| `first-batches.py`    | `[4, 4, 2]`                         |
| `batch-factory.py`    | `2 full batches; 2 samples dropped` |

### Browser, accessibility and responsiveness checklist

- [x] All three authored lesson routes, previous/next, browser back, reload and last-lesson resume.
- [x] Invalid URL fragments and stale saved lesson IDs recover to an authored lesson.
- [x] `/` and `Ctrl+K` search, highlighted results, empty state, keyboard selection, Escape and focus return. The implementation also handles `Cmd+K`.
- [x] Completion and uncompletion, bookmark save/remove, saved-lessons view, theme persistence.
- [x] Four revealable quiz answers per lesson; quick-reference and 16-block source-coverage views.
- [x] All six Copy and Download .py controls match their mirrored Python files exactly.
- [x] Actual file-URL copy success; simulated clipboard denial exercises both fallback success and truthful failure messages.
- [x] Blocked localStorage exercises the in-memory fallback without breaking navigation or state.
- [x] Interactive diagram results checked against hand-calculated values.
- [x] Section TOC follows the current section; reading progress updates with scrolling.
- [x] 360px, 768px and 1440px viewports; no document-level horizontal overflow.
- [x] Mobile drawer focus entry, reverse Tab wrap, Escape, focus restoration and navigation.
- [x] Reduced-motion layout and instant section navigation.
- [x] Automated axe WCAG A/AA scans: no violations on all three previews, dark-mode reader and search dialog.
- [x] Built document opened offline with zero remote requests and zero page errors; in-page links resolve.
- [x] Dark-mode print switches to a white page and readable light-theme body/code colors; no reader chrome. Print layout rendered and inspected.
- [x] Desktop light/dark, mobile overview and all three mobile diagrams visually inspected. Source clipping confirmed visually.

Screenshots are in `docs/screenshots/`. Useful review captures include `desktop-overview.png`, `desktop-dark-overview.png`, `mobile-overview.png`, `mobile-autograd.png`, `mobile-update.png`, and `mobile-diagram.png`. Raw browser results are in the ignored `test-results/browser-results.json`. Source and print review images are in ignored `tmp/pdf-review/`.

## Limits and remaining work

- This is intentionally the Phase 01 foundation with **three previews**, not the complete handbook. Phase 02 is next; all later content remains planned.
- Automated browser verification used Chromium on Windows. Firefox, Safari, actual mobile devices and screen-reader sessions have not been tested. Automated accessibility scans do not substitute for those manual checks.
- Browser file-origin storage/clipboard policies vary. The tested fallback behavior is implemented, but moving the HTML file may create a different storage scope.
- The temporary Python runtime did not include NumPy. PyTorch emitted an initialization warning about optional NumPy support; all six tensor-only scripts nevertheless executed successfully and require no NumPy operations.
- PDF.js emitted a PostScript-to-WASM fallback warning while rendering the source; its JavaScript fallback rendered the reviewed pages successfully.
- The host's default Python command was unavailable, and PowerShell blocked npm's `.ps1` wrapper. Workspace-local Python and `npm.cmd` were used. The temporary runtime is ignored and is not a reader dependency.
- No unresolved Phase 01 test failures remain. No source PDF edits, deployment, remote push or cloud resources were made.

# Phase 02 verification report

Verified on 25 October 2026. Phase 02 is complete; work stopped before Phase 03.

## Changeset and content

- Extended the existing typed reader with six ordered Foundations lessons: tensor essentials, derivatives and chain rule, computation DAG, scalar autograd, vector gradients, and gradient contexts. The original three Phase 01 lessons, IDs, progress behavior and search remain available.
- Read the complete extracted PDF pp. 1–5 as the primary source. The source-derived lessons cover automatic differentiation, leaves/intermediates/root, scalar backward, vector gradients, accumulation, clearing and disabled tracking.
- Added a clearly labeled supplemental tensor-prerequisites lesson because the PDF opens with autograd rather than tensor first principles. Supplemental coverage includes N-D shapes/axes, dtype/device, broadcasting, creation, indexing, reshape/view/unsqueeze/squeeze, matmul, contiguous storage and shape pitfalls. inference_mode(), finite-difference checks and additional storage guidance are identified as editorial additions.
- Added a one-screen autograd cheat sheet, shape traces, worked numerical examples, common-error guidance and four revealable questions per lesson. The updated source manifest marks Phase 02 source topics and the tensor supplement reviewed.
- Added fourteen distinct CPU-first Python examples under python_examples/02_foundations_autograd/; pytest executes each script and its assertions. No example downloads data.
- Updated the standalone reader at dist/PyTorch-Atlas.html (369,298 bytes, about 361 KiB), README, coverage generator, source manifest, snippet mirror and tests.

## Diagrams and accessibility

Five Phase 02 SVG concepts are working: tensor axes/shapes, forward/backward computation graph, interactive chain rule, vector reduction/gradients, and tracked-versus-disabled graph. The tensor-rank selector, scalar x slider and tracking checkbox update labeled conceptual values. They do not run PyTorch in the browser.

- [x] SVGs have accessible names/descriptions and accompanying prose; native sliders and checkboxes have labels and live result text.
- [x] Axe WCAG A/AA scans pass on all nine authored lessons, dark theme and search.
- [x] Offline file URL browser tests report no remote requests or page errors.
- [x] 360px, 768px and 1440px layouts have no document-level horizontal overflow; mobile drawer focus, reduced motion and print behavior pass.
- [x] Source-content and bundle checks find no video links, timecode links, or exported source chrome.

## Actual checks

| Command                                                                   | Result                                                                                                                     |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| npm.cmd run build                                                         | **PASS** — strict TypeScript build and single-file HTML packaging; JS/CSS inline                                           |
| npm.cmd test                                                              | **7 / 7 PASS** — preserved content, Phase 02 structure, 20 unique mirrored snippets, search and standalone checks          |
| npm.cmd run check:coverage                                                | **PASS** — 211 indexed pages, 16 blocks, 122 topics, 4 supplemental records                                                |
| npm.cmd run test:e2e                                                      | **17 / 17 PASS** — includes Phase 02 controls, all-lesson accessibility scan, offline file URL and responsive/print checks |
| python -m pytest -q python_examples/02_foundations_autograd/test_smoke.py | **15 / 15 PASS** — fourteen examples plus download-free inventory assertion, CPU PyTorch 2.14.1                            |
| ./tmp/python/python.exe scripts/smoke-python.py                           | **6 / 6 PASS** — all preserved Phase 01 examples still execute                                                             |

The local PyTorch runtime emits one warning because optional NumPy is absent; these examples use only PyTorch tensor operations and all assertions pass. No other Phase 02 verification issues remain. No source PDF edits, deployment or push were made.

# Phase 03 verification report

Verified on 25 October 2026. Phase 03 is complete; work stops before Phase 04.

## Delivered

- Read and visually reviewed PDF pages 6–14. Several source code lines are clipped or ambiguous in the exported chat layout, so runnable examples were reconstructed and checked against the written concepts. The source's remote dataset reference was clipped; the handbook provides deterministic synthetic data and an optional local CSV reader with no download.
- Added six ordered lessons for binary-classification setup, preprocessing, a manual linear/sigmoid model, binary cross-entropy, backward/update/gradient clearing, and evaluation. The lessons explain B=0/M=1 label direction, validation/test separation, train-only scaling, float32 tensors, `[N,30] @ [30,1] + [1]`, `[N,1]` logits/probabilities, clipping and stable BCE-with-logits, thresholding, and confusion counts.
- Added a nine-stage interactive pipeline. Selecting a stage updates its shape, equation, expected behavior, and matching Python source. A learning-rate control updates a conceptual SGD parameter/loss plot; a runnable gradient-accumulation snippet shows why gradients must be cleared.
- Added eight self-contained CPU examples under `python_examples/03_manual_training/`, including a complete classifier. Assertions cover shapes, dtypes, and loss reduction. Snippet copies are mirrored from the typed lesson registry.
- Updated module navigation, search, source references, coverage metadata, responsive styles, README, tests, and the single-file reader. The manifest now records Phase 03 as reviewed across the relevant topics and has seven supplemental records.

## Actual checks

| Command                                                                     | Result                                                                                                                                                         |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm.cmd run build`                                                         | **PASS** — strict TypeScript check and standalone HTML packaging; 407.91 kB Vite output, all assets inlined                                                    |
| `npm.cmd test`                                                              | **8 / 8 PASS** — Phase 01–03 lesson structure, all 28 unique snippets and 35 lesson snippet slots, exact Python mirrors, coverage/search and standalone checks |
| `npm.cmd run check:coverage`                                                | **PASS** — 211 pages, 16 blocks, 122 topics, 7 supplements                                                                                                     |
| `npm.cmd run test:e2e`                                                      | **18 / 18 PASS** — offline file URL, responsive/print behavior, Phase 03 explorer, and axe WCAG A/AA scans on all 15 lessons, dark theme and search            |
| `python -m pytest -q python_examples/03_manual_training/test_smoke.py`      | **10 / 10 PASS** — eight examples plus inventory and completeness checks; CPU PyTorch 2.14.1                                                                   |
| `python -m pytest -q python_examples/02_foundations_autograd/test_smoke.py` | **15 / 15 PASS** — preserved Phase 02 regression suite                                                                                                         |
| `python scripts/smoke-python.py`                                            | **6 / 6 PASS** — preserved Phase 01 examples                                                                                                                   |

The optional NumPy package is absent from the temporary PyTorch runtime, which emits a warning at initialization; all tensor-only scripts and assertions pass. The accessibility scan takes about 31 seconds across all lesson routes, so its browser-test timeout is set to 90 seconds. The reviewed pipeline capture is `docs/screenshots/phase03-pipeline.png`. No source PDF edits, deployment, or push were made.

## Phase boundary

Phase 04 is next: `nn.Module`, `nn.Sequential`, and `Dataset`/`DataLoader`. The reader now has fifteen authored lessons and retains all Phase 01–03 routes and progress IDs.

# Phase 04 verification report

Verified on 25 October 2026. Phase 04 is complete; work stops before Phase 05.

## Delivered

- Read PDF pp. 15–40 in full and visually reviewed the rendered pages. Several exported code lines and the source CSV URL are clipped; examples were reconstructed and executed without copying the broken remote address.
- Added ten lessons comparing the manual Phase 03 classifier with idiomatic PyTorch; teaching `nn.Module`, `super().__init__()`, `nn.Linear`, `nn.Parameter`, hooks, `model(x)`, `parameters()` and `state_dict()`; explaining `Sequential` vs explicit `forward`, activations, optional `torchinfo`, loss/output contracts and SGD lifecycle; and covering custom/TensorDataset, samplers, collation/padding, loader options, the full batched classifier and safe evaluation/checkpoint patterns.
- Added eighteen distinct standalone CPU snippets (22 lesson code slots). They include side-by-side manual-vs-SGD, manual class-vs-module and Dataset-vs-DataLoader examples; a stable logits-first classifier; shape/dtype error reproductions; custom and tensor datasets; explicit RandomSampler and SequentialSampler examples; padded variable-length text; portable worker guidance; a complete leakage-aware train/validation/test loader pipeline; and a weights-only state-dict round trip.
- Added four accessible SVG concepts for module parameter registration, optimizer lifecycle, sampler-to-model loading and padded collation. The existing ten-record batch-size slider continues to show batch counts and `drop_last`; it covers sizes 1–5.
- Updated navigation, quick reference/search, source coverage, snippet mirrors, offline build and tests. The manifest now marks Phase 04 source topics reviewed and records twelve supplemental items, including logits-first loss guidance, validation/test separation, safe checkpoint loading, Windows workers, conditional pinned memory and iterable-stream caveats.

## Actual checks

| Command                                                                     | Result                                                                                                                                                                                   |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm.cmd run build`                                                         | **PASS** — strict TypeScript build and standalone HTML; 469.93 kB Vite output, assets inlined                                                                                            |
| `npm.cmd test`                                                              | **9 / 9 PASS** — 25 authored lessons, 57 lesson snippet slots, 46 unique snippets and exact mirrors                                                                                      |
| `npm.cmd run check:coverage`                                                | **PASS** — all 211 pages, 16 blocks, 122 topics, 12 supplements                                                                                                                          |
| `npm.cmd run test:e2e`                                                      | **19 / 19 PASS** — file URL offline, no remote requests/errors, Phase 04 diagrams/download, responsive and print checks; axe WCAG A/AA scans across all 25 lessons, dark mode and search |
| `python -m pytest -q python_examples/04_modules_data/test_smoke.py`         | **20 / 20 PASS** — eighteen examples and two inventory/completeness checks; CPU PyTorch 2.14.1                                                                                           |
| `python -m pytest -q python_examples/03_manual_training/test_smoke.py`      | **10 / 10 PASS** — Phase 03 regression                                                                                                                                                   |
| `python -m pytest -q python_examples/02_foundations_autograd/test_smoke.py` | **15 / 15 PASS** — Phase 02 regression                                                                                                                                                   |
| `python scripts/smoke-python.py`                                            | **6 / 6 PASS** — Phase 01 regression                                                                                                                                                     |

The full synthetic minibatch example reports training loss 0.749 → 0.185 and 0.90 validation and test accuracy. PyTorch warns that optional NumPy is absent in the temporary runtime; the examples use tensor operations and pass their assertions. The optional `torchinfo` summary is guarded and reports a direct parameter count when the dependency is unavailable. PDF rendering uses its JavaScript fallback after a PostScript-to-WASM warning. No source PDF edits, deployment or push were made.

The parameter-registration SVG was visually inspected at `docs/screenshots/phase04-module-tree.png`; its labels and registered parameter shapes are legible at full size.

## Phase boundary

Phase 05 is next: Fashion-MNIST, ANN image classification and device placement. The reader now has twenty-five authored lessons and retains the earlier routes and progress IDs.

# Phase 05 verification report

Verified on 25 October 2026. Phase 05 is complete; work stops before Phase 06.

## Delivered

- Read PDF pp. 41-59 in full and visually reviewed the rendered pages. The exported ANN and GPU code blocks are clipped or incomplete, so the exercises were reconstructed with explicit tensor contracts and safe CPU paths.
- Added five authored lessons on Fashion-MNIST representation and split flow; a 784-128-64-10 MLP; multiclass training and evaluation modes; training/validation curves and confusion-matrix metrics; and CUDA/MPS/CPU device placement.
- Added ten distinct Phase 05 snippets (12 lesson code slots), including a synthetic multiclass training run, shape and dtype contracts, parameter accounting, weighted metrics, a safe state-dict round trip, train/validation history, and device selection. The separate torchvision Fashion-MNIST exercise defaults to `DOWNLOAD_DATA = False`; set it to `True` explicitly to fetch missing files.
- Added five accessible SVG concepts: a hidden-width ANN control that recalculates parameters, dataset split/flatten flow, device selection and placement, a confusion matrix, and training/evaluation flow. The ANN defaults to 109,386 trainable parameters and recalculates correctly when widths change.
- Marked Phase 05 source topics reviewed and added six supplemental records on logits, validation-based selection, balanced metrics, backend fallback, transfer guidance and opt-in dataset downloads. Navigation/search, source references, mirrors, README and the offline package now include Phase 05.

## Actual checks

| Command                                                        | Result                                                                                                                                                           |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `npm.cmd run build`                                            | **PASS** - strict TypeScript build and self-contained HTML; 505.62 kB Vite output, 494 KB packaged                                                               |
| `npm.cmd test`                                                 | **10 / 10 PASS** ? 30 authored lessons, 72 lesson snippet slots, 56 unique snippets, exact mirrors                                                               |
| `npm.cmd run check:coverage`                                   | **PASS** - all 211 pages, 16 blocks, 122 topics, 18 supplements                                                                                                  |
| `npm.cmd run test:e2e`                                         | **20 / 20 PASS** ? offline file URL, no remote requests/errors, ANN controls, diagrams/downloads, responsive/print and axe WCAG A/AA scans across all 30 lessons |
| `python -m pytest -q python_examples/05_ann_gpu/test_smoke.py` | **11 / 11 PASS** ? ten CPU examples plus inventory and opt-in-download checks                                                                                    |
| Phase 04 Python smoke suite                                    | **20 / 20 PASS**                                                                                                                                                 |
| Phase 03 Python smoke suite                                    | **10 / 10 PASS**                                                                                                                                                 |
| Phase 02 Python smoke suite                                    | **15 / 15 PASS**                                                                                                                                                 |
| `python scripts/smoke-python.py`                               | **6 / 6 PASS**                                                                                                                                                   |

The synthetic ANN example confirms training loss falls and all 50 validation labels are counted. CPU fallback was exercised; CUDA and MPS hardware were not available for runtime verification. The separate torchvision script was not executed because it is intentionally optional and may download data only after the user changes its flag. PyTorch reports that optional NumPy is absent in the temporary runtime; the tensor-only tests pass. No source PDF edits, deployment or push were made.

## Phase boundary

Phase 06 is next: regularization and generalization on PDF pp. 60-71. The reader now has thirty authored lessons and retains all earlier routes and progress IDs.

# Phase 06 verification report

Verified on 25 October 2026. Phase 06 is complete; work stops before Phase 07.

## Delivered

- Read PDF pp. 60-88 completely and visually reviewed the rendered pages. Exported training and Optuna cells are clipped; examples were reconstructed. Source accuracy figures are presented as reported observations, not guarantees.
- Added fifteen lessons in order covering generalization and train/validation/test protocol; dropout; BatchNorm1d; L2/weight decay and AdamW; combined regularization and SGD/momentum/Adam/AdamW; search strategies; Optuna studies, trials, objectives and samplers; leakage-safe tabular CV; sampler comparison; study plots; conditional search, persistence, callbacks/pruners; the eight ANN settings; dynamic model construction; and validation-only ANN tuning.
- Added ten distinct Python snippets (49 lesson code slots) and mirrored them in `python_examples/06_regularization_optuna/`. Eight CPU-executable examples cover mode behavior, BatchNorm numerics, shrink-only SGD, optimizer setup, synthetic ANN training, curves, local experiment logs and lazy plotting. Two Optuna study scripts are optional and clearly identify the dependency. The ANN objective returns validation loss, refits its selected setup, and accesses test only for the final report.
- Added seven accessible SVG explorers for generalization curves, dropout masks, BatchNorm gamma/beta, weight decay, optimizer paths, search spaces and trial comparisons. Controls update conceptual values; optimizer and study diagrams label schematic observations as illustrative.
- Updated the coverage manifest through p.88 and added seven supplemental records. The reader registry now contains 45 lessons, with previous lesson IDs and routes preserved. README and coverage details describe Phase 07 as next.

## Actual checks

| Command                                                                      | Result                                                                                                                     |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `npm.cmd run build`                                                          | **PASS** - strict TypeScript and single-file build; 585.59 kB Vite output, 572 KB packaged                                 |
| `npm.cmd test`                                                               | **11 / 11 PASS** - 45 authored lessons, 121 lesson code slots, 66 unique snippets and exact mirrors                        |
| `npm.cmd run check:coverage`                                                 | **PASS** - all 211 pages, 16 blocks, 122 topics, 25 supplements                                                            |
| `npm.cmd run test:e2e`                                                       | **21 / 21 PASS** - offline reader, all-lesson axe WCAG A/AA scan, Phase 06 controls, downloads and responsive/print checks |
| `python -m pytest -q python_examples/06_regularization_optuna/test_smoke.py` | **10 / 10 PASS** - eight CPU examples plus optional-dependency/leakage and inventory checks                                |
| Phase 05 Python smoke suite                                                  | **11 / 11 PASS**                                                                                                           |
| Phase 04 Python smoke suite                                                  | **20 / 20 PASS**                                                                                                           |
| Phase 03 Python smoke suite                                                  | **10 / 10 PASS**                                                                                                           |
| Phase 02 Python smoke suite                                                  | **15 / 15 PASS**                                                                                                           |
| `python scripts/smoke-python.py`                                             | **6 / 6 PASS**                                                                                                             |

Representative diagrams were visually reviewed at `docs/screenshots/phase06-dropout.png`, `phase06-batchnorm.png` and `phase06-search-space.png`. CUDA and MPS were not needed for these synthetic examples. Optuna is not installed in the validation runtime, so its actual study execution and Plotly rendering were not run; those scripts remain optional and the dependencies are named in their snippet labels. The synthetic Optuna code is not represented as verified execution. No source PDF edits, deployment or push were made.

## Phase boundary

Phase 07 is complete below; Phase 08 is next.

# Phase 07 verification report

Verified on 25 October 2026. Phase 07 is complete; work stops before Phase 08.

## Delivered

- Read PDF pp. 89-121 in full. Added nine ordered lessons covering CNN motivation and locality, NCHW images, cross-correlation, output-size math, channels and parameters, ReLU/BatchNorm2d, pooling, receptive fields, and the complete Fashion-MNIST CNN.
- Added fifteen unique snippet scripts (including templates): fourteen CPU examples and one separately guarded torchvision lab with the download flag disabled. Examples cover manual convolution versus F.conv2d, output formula versus PyTorch shapes, multichannel filters, pooling, receptive field, complete model, hooks-based shape trace, parameter count, synthetic training and debugging.
- Added seven original accessible SVG views: locality, sliding kernel with live arithmetic, padding/stride shape control, channel/filter parameter control, max/average pooling, receptive-field progression and full CNN flow.
- Recalculated the feature extractor count as 19,008 including BatchNorm affine parameters; full model count is 429,450. Clarified flattened values remain present, PyTorch uses cross-correlation, and CNNs do not guarantee general translation invariance.
- Updated source mappings through p.121, mirrors, navigation, README and source audit. Seven supplemental records are marked separately.

## Checks

| Command | Result |
| --- | --- |
| npm.cmd run build | PASS — strict TypeScript and one-file offline HTML |
| npm.cmd test | PASS — content structure, routes, snippet mirrors and source-chrome checks |
| npm.cmd run check:coverage | PASS — 211 pages, 16 blocks, 122 topics, 19 supplements (7 added in Phase 07) |
| python -m pytest -q python_examples/07_convolutional_networks/test_smoke.py | PASS — fourteen CPU examples plus optional-download and mirror checks |
| npm.cmd run test:e2e | PASS — 22 / 22 tests, including offline file URL, all-lesson axe scan and responsive checks |

Browser checks cover all 54 authored lessons, WCAG A/AA axe scans, interactive CNN values/shapes/counts, offline file URL, and responsive/print behavior. Optional real Fashion-MNIST training was not run because it intentionally requires an explicit download and torchvision. CNN accuracy is not asserted from synthetic data.

## Phase boundary

Phase 08 is complete below; Phase 09 is next. No source PDF edits, deployment or push were made.

# Phase 08 verification report

Verified on 25 October 2026. Phase 08 is complete; work stops before Phase 09.

## Delivered

- Read PDF pp. 121–141 in full. Added seven lessons: transfer learning, frozen feature extraction vs fine-tuning, CSV grayscale-to-RGB conversion, resize/crop/tensor conversion, ImageNet normalization, VGG16 classifier/training flow, and the weights/transforms API.
- Added nine distinct scripts (22 lesson code slots including templates/additional examples). Eight execute without downloads; one full pretrained VGG16 lab is gated by ALLOW_DOWNLOADS=False and uses a train/validation split with final test evaluation.
- Added four accessible interactive SVG diagrams: transfer stages, preprocessing pipeline, freeze/fine-tune strategy, and channel normalization arithmetic.
- Reconstructed shape and dtype steps, verified the normalization example, and used VGG16_Weights.DEFAULT with its paired transform. The full lab converts Fashion-MNIST grayscale PIL images to RGB before applying the ImageNet preset.
- Updated source mappings through p.141 and added three supplemental records for validation-based selection, weight-paired transforms, and explicit downloads. Clarified the source code is feature extraction, not fine-tuning.

## Checks

| Command | Result |
| --- | --- |
| npm.cmd run build | PASS — strict TypeScript and self-contained HTML |
| npm.cmd test | PASS — 13 / 13 content checks |
| npm.cmd run check:coverage | PASS — 211 pages, 16 source blocks, 122 topics, 22 supplements |
| python -m pytest -q python_examples/08_transfer_learning/test_smoke.py | PASS — 8 offline examples plus download-gate and mirror checks (10 / 10) |
| npm.cmd run test:e2e | PASS — 23 / 23 tests; offline file URL, all-lesson axe scan, transfer controls, and responsive layouts |

Browser coverage includes the seven new lessons in the all-lesson axe scan; targeted controls are checked separately. The full pretrained training lab was not executed because it downloads ImageNet weights and Fashion-MNIST and requires substantial compute. CUDA was not required for smoke checks.

## Phase boundary

Phase 09 covers recurrent neural networks beginning at PDF p.142. No source PDF edits, deployment, or push were made.

# Phase 09 verification report

Verified on 25 October 2026. Phase 09 is complete; work stops before Phase 10.

## Delivered

- Read PDF pp. 142–167 in full. Added seven ordered lessons covering sequence motivation and use cases, input axes, shared recurrent cell math and parameter counts, output patterns, BPTT and gradient pathologies.
- Added ten distinct CPU-safe snippets (14 lesson code/template slots), mirrored under `python_examples/09_recurrent_networks/`. They verify recurrent shapes, PyTorch's 34-parameter example versus the source's Keras 31, manual cell arithmetic, binary and per-step classifiers, packed variable lengths, finite BPTT gradients, scalar gradient products, norm clipping and a train/eval update.
- Added five accessible interactive SVG diagrams for sequence axes, recurrent unrolling, task output patterns, BPTT and gradient products. Controls update the highlighted step, architecture mapping or scalar illustration; no Python executes in the browser.
- Corrected the Keras/PyTorch bias-count comparison and clarified tanh derivatives and Jacobian products. Labeled embeddings, PackedSequence, PyTorch autograd workflow and clipping guidance as supplemental. The FAQ classifier on pp.168–179 remains for Phase 10.
- Updated navigation, content checks, offline snippet mirrors, source audit, README and source coverage. All 11 source topics covering pp.142–167 are reviewed; the manifest has 27 supplemental entries.

## Checks

| Command | Result |
| --- | --- |
| `npm.cmd run build` | PASS — TypeScript and one-file offline HTML (700 KB) |
| `npm.cmd test` | PASS — 14 / 14 content checks, including exact Python mirror checks |
| `npm.cmd run check:coverage` | PASS — 211 pages, 16 blocks, 122 topics, 27 supplements |
| `python -m pytest -q python_examples/09_recurrent_networks/test_smoke.py` | PASS — ten CPU examples plus registry check (11 / 11) |
| `npm.cmd run test:e2e` | PASS — 24 / 24 tests, including all-lesson axe scans, offline file URL, RNN controls and responsive layouts. |

No real dataset training was required; the toy snippets demonstrate tensor and autograd mechanics, not sequence-model quality. CUDA and downloads were not used.

## Phase boundary

Phase 10 covers the FAQ question-answering classifier on PDF pp.168–179. No source PDF edits, deployment or push were made.

# Phase 10 verification report

Verified on 25 October 2026. Phase 10 is complete; work stops before Phase 11.

## Delivered

- Read PDF pp. 168–179 in full. Added seven lessons on the FAQ project scope, text normalization, PAD/UNK vocabulary, distinct answer-class labels, Dataset/DataLoader batching, embedding/RNN tensor flow, cross-entropy training and inference/checkpoints.
- Added twelve unique CPU-safe snippets (15 lesson code/template/additional slots) mirrored under `python_examples/10_faq_classifier/`. They cover tokenization, vocabulary and OOV handling, Dataset items, variable-length collation, model shapes, packed lengths, loss/target contracts, abstention heuristics, source target reconstruction, safe checkpoint mappings and a complete train/validation/inference/save/restore run.
- Added six accessible SVG diagrams: question tokens, special vocabulary IDs, padded batches, embedding/RNN classification, probability interpretation and a fixed illustrative FAQ selector. The selector uses hand-authored example mappings, explicitly says it is not a PyTorch prediction, and refuses to guess for its unknown example.
- Reconstructed the clipped sample code as a local synthetic dataset. The lessons retain the source-reported roughly 90 FAQ pairs, vocabulary size 324 for that dataset, embedding size 50, hidden size 64, Adam/CrossEntropyLoss loop, and 0.5 maximum-softmax threshold as source-specific details. The original `answer[0]` target predicts one vocabulary token; the supplemental repair predicts a fixed answer class and retrieves its full stored response.
- Added six supplemental coverage records: answer-class mapping, train-only vocabulary with separate PAD/UNK, variable-length collation, packed sequences, calibrated-abstention caveat, and synthetic validation/checkpoint handling. FAQ source topics are reviewed; the coverage manifest has 33 supplemental records.

## Checks

| Command | Result |
| --- | --- |
| `npm.cmd run build` | PASS — TypeScript and one-file offline HTML (750 KB) |
| `npm.cmd test` | PASS — 15 / 15 content checks and exact snippet mirror checks |
| `npm.cmd run check:coverage` | PASS — 211 pages, 16 blocks, 122 source topics, 33 supplements |
| `python -m pytest -q python_examples/10_faq_classifier/test_smoke.py` | PASS — twelve CPU snippets plus registry check (13 / 13) |
| `npm.cmd run test:e2e` | PASS — 25 / 25 browser tests, including offline file URL, all-lesson axe scan and responsive layouts |
| Final-file targeted browser checks | PASS — all-lesson WCAG A/AA scan and FAQ diagram selector on the final build |

The FAQ lab uses six synthetic answer classes and held-out paraphrases only to exercise the pipeline. It does not establish FAQ generalization or provide a calibrated OOD detector. No external dataset, CUDA or download was used.

## Phase boundary

Phase 12 covers next-word prediction and begins at PDF p.198. No source PDF edits, deployment or push were made.

# Phase 11 verification report

Verified on 26 October 2026. Phase 11 is complete; work stops before Phase 12.

- Read the prompt pack and source pp.180–198. Added nine ordered LSTM lessons: motivation, state roles, vector/gate inputs, gate equations, blueprint, worked numerical example, gradient-route limits, PyTorch shape contracts and parameter counting.
- The seven concept lessons are source-derived; PyTorch API shapes and native parameter accounting are labeled supplemental. Coverage manifest tracks 123 source topics and 39 supplemental items.
- Added 12 CPU-safe snippets (11 unique IDs, with intentional reuse of the same numerical example), including a numerical gate update, equivalence check against `nn.LSTMCell`, bidirectional shape/parameter checks, packed lengths, state carry and gradient-path calculations.
- Added six authored SVG diagram kinds: dual state, vectors, interactive cell arithmetic, gradient route, tensor shapes and parameter counts. They have labels, descriptions and the shared value/gradient legend.
- Source nuance: sigmoid gate endpoints are idealized; the direct cell path contains forget-gate products but is not the full Jacobian. The LSTM is not represented as a guarantee of indefinite memory.

| Command | Result |
|---|---|
| `npm.cmd run build` | PASS — TypeScript and standalone `dist/PyTorch-Atlas.html` (799 KB); bundled assets only |
| `npm.cmd test` | PASS — 16 content tests, 84 lessons, 216 lesson code slots and 123 unique snippets |
| `npm.cmd run check:coverage` | PASS — 211 pages, 16 blocks, 123 source topics, 39 supplemental records |
| `python -m pytest -q python_examples/11_lstm_internals/test_smoke.py` | PASS — 12 snippets, CPU only |
| `npm.cmd run test:e2e` | PASS — 26 browser tests including offline file URL, LSTM controls, responsive/print and WCAG scans across all 84 lessons |

Accessibility/responsiveness checklist: SVGs have meaningful accessible labels and text descriptions; slider controls have labels and live results; shared colors and legend preserve the atlas theme; keyboard focus and reduced-motion behavior inherit the reader shell; existing 360/768/1440 px responsive and print scans pass.

Unresolved limitations: the direct forget product is a deliberately simplified derivative route; a full LSTM Jacobian depends on gate derivatives and recurrent paths. The synthetic classifier and hand-computed gates teach mechanics and are not performance benchmarks. Phase 12 next-word modeling remains unimplemented.

# Phase 12 verification report

Verified on 26 October 2026. Phase 12 is complete; the requested phase sequence now covers PDF pp. 1–211.

## Delivered

- Read and audited PDF pp. 198–211 in full, alongside the prior phase review of pp. 180–198. Added ten ordered lessons: dataset/scope, tokenization, train-only vocabulary, causal prefixes, left padding, LSTM architecture, training, greedy generation, top-k/temperature sampling and the complete project. Nine are source-derived; sampling and its controls are explicitly supplemental.
- Added fourteen CPU-safe runnable Phase 12 snippets, mirrored in `python_examples/12_lstm_next_word/`. The consolidated CLI trains and saves a reproducible model, then loads it for greedy or top-k generation. It uses a tiny inline corpus by default; a local UTF-8 corpus is opt-in. The split happens at sentence level before prefixes, and the vocabulary is fit on training sentences only.
- Added five accessible interactive SVGs for causal prefixes, left padding, tensor shapes, illustrative sampling and autoregressive append steps. All controls illustrate fixed conceptual state; the browser does not execute Python.
- Added coverage review for all source topics and supplements, plus nine grouped source-correction entries. The reader now exposes the correction register and source coverage index.
- Corrected a stale Phase 11 mirrored assertion discovered by cross-phase CPU smoke checks, then resynchronized the canonical snippet registry and Python mirrors.

## Checks

| Command | Result |
| --- | --- |
| `npm.cmd run snippets:sync` | PASS — 236 lesson code slots mirrored |
| `npm.cmd test` | PASS — 18 content tests, including lesson, registry and source-manifest checks |
| `npm.cmd run check:coverage` | PASS — 211 pages, 16 blocks, 124 reviewed source topics, 45 reviewed supplements and 9 correction groups |
| `npm.cmd run build` | PASS — standalone `dist/PyTorch-Atlas.html` (about 855 KB) |
| `python scripts/smoke-python.py` | PASS — six Phase 1 examples |
| Phase 2–12 `test_smoke.py` suites, run one folder at a time | PASS — 121 tests total, CPU only |
| `npm.cmd run test:e2e` | PASS — 27 / 27 browser tests, including offline `file://`, all-lesson accessibility scans, responsive layouts, print, reduced motion and Phase 12 copy/download controls |

Accessibility/responsiveness checklist: the five new SVGs have accessible labels and descriptions; diagrams have labeled controls and visible explanatory text; shared keyboard focus, dark/light themes and reduced-motion support remain in effect; browser scans cover 360, 768 and 1440 px layouts and print output.

Unresolved limitations: the inline corpus is intentionally tiny and cannot establish language quality. Tokenization is a small regex-based teaching implementation rather than NLTK. The combined `python -m pytest -q python_examples` invocation cannot collect repeated `test_smoke.py` module names across folders; all 121 phase tests passed when invoked per folder. No GPU, hidden download, deployment or push was used.

## Phase boundary

Phase 12 completes the current prompt pack through PDF p. 211. No source PDF edits, deployment or push were made.
