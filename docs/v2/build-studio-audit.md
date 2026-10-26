# Build Studio · BS01–BS05 implementation record

## Project inventory

The F16 codebase had the Field Kit and 94 lessons but no Build Studio routes, project schema, or project scripts. F17 adds a typed `BuildProject` contract, statically bundled Python script registry, searchable project index, responsive project reader, staged snippet stepper, schema and shape tables, copy/download controls, source backlinks, and 15 accessible SVG visualizations.

Each project includes a problem and stakeholder, target/metric, data provenance, sample record schema, baseline, architecture reasoning, split and fit boundaries, shape trace, six staged code examples, a complete Python script, an offline smoke test, run instructions, expected contracts, debugging notes, extension work, and retrieval questions.

## Source mapping

| Project | Classification | Source pages | Evidence / qualification |
|---|---|---:|---|
| BS01 House price regression | Supplemental | — | Seeded numeric teaching rule; not housing data or a market estimate. |
| BS02 Wisconsin breast cancer | Source-derived + labeled repairs | PDF pp. 6–14 | Source uses Wisconsin diagnosis features and a binary pipeline. Current exercise uses the built-in scikit-learn dataset, `nn.Module`, float32, BCEWithLogitsLoss, training-only scaling, validation threshold selection, and an explicit nonclinical warning. |
| BS03 Customer churn ANN | Supplemental | — | Seeded synthetic labels; not a real company or representative churn data. Optuna is an optional validation-only extension. |
| BS04 Fashion-MNIST ANN | Source-derived + supplemental setup | PDF pp. 41–49 | Preserves 28×28 grayscale data, flattening, 784→128→64→10, ReLU, logits and cross-entropy. Source performance figures remain qualified as reported examples. |
| BS05 Fashion-MNIST CNN | Source-derived + supplemental setup | PDF pp. 89–97 | Preserves local convolution, two Conv/Pool blocks, BatchNorm2d, classifier and 28→14→7 shape path. Reconstructs the clipped CSV loader with an explicit optional dataset flag; source accuracy figures are not guarantees. |

The PDF was inspected through the local page text for pp. 6–14, 41–49 and 89–97. Exported code with clipped loader and convolution lines was reconstructed instead of copied. The per-project mapping is machine-readable in `docs/source-coverage.json` under `buildProjects`.

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

The checkout is on the pushed F16 branch, while F16 is not merged into `main`; F17 continues from that branch tip so it retains the just-added Field Kit. This work stops at F17 and does not add BS06–BS10.

## Known limits

The browser illustrations are authored explanations, not measured model plots. Fashion-MNIST training is opt-in because it downloads data; its default synthetic smoke checks code contracts only. BS01 and BS03 use synthetic data. The local CSV mode in BS01 expects the documented numeric columns and does not infer arbitrary schemas.
