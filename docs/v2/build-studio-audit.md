# Build Studio Â· BS01â€“BS05 implementation record

## Project inventory

The F16 codebase had the Field Kit and 94 lessons but no Build Studio routes, project schema, or project scripts. F17 adds a typed `BuildProject` contract, statically bundled Python script registry, searchable project index, responsive project reader, staged snippet stepper, schema and shape tables, copy/download controls, source backlinks, and 15 accessible SVG visualizations.

Each project includes a problem and stakeholder, target/metric, data provenance, sample record schema, baseline, architecture reasoning, split and fit boundaries, shape trace, six staged code examples, a complete Python script, an offline smoke test, run instructions, expected contracts, debugging notes, extension work, and retrieval questions.

## Source mapping

| Project | Classification | Source pages | Evidence / qualification |
|---|---|---:|---|
| BS01 House price regression | Supplemental | â€” | Seeded numeric teaching rule; not housing data or a market estimate. |
| BS02 Wisconsin breast cancer | Source-derived + labeled repairs | PDF pp. 6â€“14 | Source uses Wisconsin diagnosis features and a binary pipeline. Current exercise uses the built-in scikit-learn dataset, `nn.Module`, float32, BCEWithLogitsLoss, training-only scaling, validation threshold selection, and an explicit nonclinical warning. |
| BS03 Customer churn ANN | Supplemental | â€” | Seeded synthetic labels; not a real company or representative churn data. Optuna is an optional validation-only extension. |
| BS04 Fashion-MNIST ANN | Source-derived + supplemental setup | PDF pp. 41â€“49 | Preserves 28Ã—28 grayscale data, flattening, 784â†’128â†’64â†’10, ReLU, logits and cross-entropy. Source performance figures remain qualified as reported examples. |
| BS05 Fashion-MNIST CNN | Source-derived + supplemental setup | PDF pp. 89â€“97 | Preserves local convolution, two Conv/Pool blocks, BatchNorm2d, classifier and 28â†’14â†’7 shape path. Reconstructs the clipped CSV loader with an explicit optional dataset flag; source accuracy figures are not guarantees. |

The PDF was inspected through the local page text for pp. 6â€“14, 41â€“49 and 89â€“97. Exported code with clipped loader and convolution lines was reconstructed instead of copied. The per-project mapping is machine-readable in `docs/source-coverage.json` under `buildProjects`.

## Visualization provenance

All 15 browser visuals are fixed illustrative SVGs, not model output. BS01: feature flow, regression fit intuition, residual axis. BS02: feature/logit path, interactive conceptual threshold, empty confusion-count layout. BS03: synthetic feature panel, ANN shape path, conceptual threshold. BS04: pixel-to-vector grid, layer widths, class-score slots. BS05: local kernel footprint, feature-map stages, exact shape journey. There are no measured charts bundled. The explicit Fashion-MNIST full training scripts print metrics measured by that local run and do not claim benchmark performance.

## Local smoke scripts

Each `python_projects/bsXX_*/train.py` has an offline default. BS02 uses only scikit-learn's bundled dataset. BS04/BS05 import torchvision only in the explicit full-training branch and require both `--download-fashion-mnist --train-full`. Validation metrics and any threshold choice use validation data; smoke paths do not inspect test labels. Fashion ANN and CNN share the same seed-17 train/validation split function and official test set identity.

## Verification checklist

- [x] Responsive project index and typed project reader
- [x] Search includes Build Studio projects and routes to project hashes
- [x] Lesson and Field Kit backlinks resolve through the existing router
- [x] Copy/download complete code uses the same text as the `.py` source
- [x] Every visual has title, description, visible caption, and provenance label
- [x] Narrow-screen layout and dark theme rules added
- [x] TypeScript and production build pass; `dist/PyTorch-Atlas.html` is generated as one offline HTML file
- [x] `npm.cmd test`: 26/26 content tests pass
- [x] `npm.cmd run check:coverage`: 211 pages, 16 blocks, 124 reviewed topics, 45 supplements, and 9 correction groups pass; project source mappings pass
- [x] `python -m pytest python_projects -q`: 6/6 CPU-safe smoke tests pass
- [x] Each of the five project CLI defaults completes with synthetic or bundled local data; no hidden downloads
- [x] `npm.cmd run verify:dist`: GitHub Pages entry and standalone package are identical and self-contained
- [x] Focused Build Studio and Field Kit browser accessibility checks pass, including 390px layout
- [x] `npm.cmd run test:e2e`: 34/34 browser cases pass (including the 94-lesson axe scan, dark mode, search, file URL, responsive views, and project navigation)

## Adaptation from plan

The checkout is on the pushed F16 branch, while F16 is not merged into `main`; F17 continues from that branch tip so it retains the just-added Field Kit. This work stops at F17 and does not add BS06â€“BS10.

## Known limits

The browser illustrations are authored explanations, not measured model plots. Fashion-MNIST training is opt-in because it downloads data; its default synthetic smoke checks code contracts only. BS01 and BS03 use synthetic data. The local CSV mode in BS01 expects the documented numeric columns and does not infer arbitrary schemas.

# Build Studio Â· F18 implementation record

## Project inventory and source audit

BS06â€“BS10 are integrated as typed project records, with three accessible SVGs apiece, staged explanations, executable scripts, local smoke tests, search, breadcrumbs, completion navigation, copy/download controls and lesson/Field Kit links. Each SVG includes a title, description, visible caption, and an Illustrative provenance label. The fallback flow layout scales to projects with six labels so the diagrams stay within the viewBox.

| Project | Classification | Source pages | Qualification |
|---|---|---:|---|
| BS06 VGG16 flower classifier | PDF method + supplemental domain | PDF pp. 121â€“141 | VGG transfer, freezing and weights.transforms() are source-derived. Flower classes are user supplied; no dataset is bundled. |
| BS07 SMS spam RNN | Supplemental | â€” | Synthetic messages by default, optional local CSV. No external SMS dataset is fetched. |
| BS08 FAQ intent classifier | PDF-derived with repair | PDF pp. 168â€“179 | Finite intent logits map to full stored answers. Explicitly a toy FAQ router, not general question answering. |
| BS09 Energy forecasting | Supplemental | â€” | Seeded synthetic daily/weekly series; chronological targets and train-only scaling. No utility data is bundled. |
| BS10 Next-word LSTM | PDF-derived with repairs/extensions | PDF pp. 198â€“211 | Offline tokenizer, sentence split, distinct PAD/UNK/EOS IDs, bounded top-k decoding and saved mappings. Tiny corpus is not a useful language model. |

The relevant PDF page text was read in full before implementing these projects. Coverage and classification are recorded per project in `docs/source-coverage.json`. PDF pages for FAQ and next-word also map to reviewed lesson topics. No source PDF changes were made.

## Verification

- [x] `npm.cmd run projects:sync` bundles all 10 scripts statically.
- [x] `npm.cmd test`: 26/26 content checks pass, including script mirrors and project links.
- [x] `npm.cmd run check:coverage`: all 211 pages, 16 blocks, 124 reviewed topics, 45 supplements, 9 correction groups and 10 project mappings pass.
- [x] `python -m pytest -q python_projects`: 18 CPU/offline tests pass across all ten projects.
- [x] Browser checks cover multiword project search, FAQ provenance labels, accessible diagram names, responsive layout, offline file navigation and the existing full reader flows.
- [x] Standalone package contains no remote asset references or runtime JSON fetches.

BS06 default smoke runs without importing torchvision or requesting ImageNet weights. BS07 and BS09 use deterministic toy data; optional local input paths are explicit. BS08 reuses the existing reconstructed FAQ script. BS10 mirrors the audited local next-word script.

## Known limits

The optional real-data runs for VGG16, SMS, and energy are not benchmark claims. SMS CSV training uses a stratified train/validation/test split and a train-only vocabulary; the seed corpus remains too small for performance claims. Energy CSV training parses sorted timestamps, requires complete hourly cadence, and runs the same chronological train-only scaling/window pipeline as the synthetic path. The BS07 toy split is intentionally small and fixed; BS08 likewise uses a tiny paraphrase split. The synthetic data cannot establish real-domain generalization. ImageNet pretrained weights and user image directories were not fetched or trained during offline verification.
