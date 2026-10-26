# PyTorch Atlas

An offline technical textbook built from the source PDF in this directory. **Phases 01–12 are implemented.** The reader preserves three previews and adds tensor/autograd, manual classification, idiomatic modules, data loading, image classification, regularization/Optuna, CNNs, VGG16 transfer learning, recurrent networks, a finite-class FAQ demonstration, LSTM internals and a complete local next-word LSTM project. The original PDF is kept separate from the distributed reader.

## Overview

PyTorch Atlas is an interactive learning and revision handbook covering PyTorch foundations, neural networks, CNNs, RNNs, LSTMs, and practical implementation. It contains 94 lessons developed across 12 phases.

## Features

- Search across lesson content, keyboard navigation, responsive module sidebar, breadcrumbs and previous/next lessons.
- Interactive, authored SVG diagrams; accessible mathematical explanations; revealable revision questions.
- Copyable Python snippets with mirrored runnable scripts, optional downloads, and clear local/Colab instructions. Python does not run in the browser.
- Bookmarks, completion progress, resume position, reading progress and light/dark themes, persisted locally where storage is available.
- One-file offline handbook at `dist/PyTorch-Atlas.html`.

## Technology stack

React 19 and TypeScript 5 are bundled by Vite 7 with `vite-plugin-singlefile`. Styling is local CSS with design tokens; diagrams use authored SVG/React; code highlighting uses PrismJS. There are no runtime API calls or CDN assets.

## Open the offline reader

Open [`dist/PyTorch-Atlas.html`](dist/PyTorch-Atlas.html) directly in a modern browser. No server, installation, account, API key, or network connection is needed to read it. The one HTML file is the complete distribution artifact; you can copy it to another computer.

Use `/` or `Ctrl/Cmd+K` to search. Save lessons, mark completion, change theme, and return to the last lesson. These preferences use browser storage on the current device; when storage is blocked they fall back to session memory. File-origin storage policy varies by browser, so moving the file can change which saved state is available.

Python examples do **not** run inside the HTML. Copy them into a Python file or a Colab Python cell with PyTorch installed. Phase 02–12 core examples use CPU tensors and synthetic/local data; no downloads are needed for the next-word project. Optional Optuna studies require the separate optuna dependency. Optional CNN and VGG16 Fashion-MNIST training require torchvision and explicit download-flag changes; VGG16 also fetches pretrained weights. Optional CSV readers accept a user-provided local file and never download it. Copy has a permission-aware fallback and reports failure when neither path is permitted; selecting code and copying manually remains possible.

## Getting started

Node 22.12+ or Node 24 and npm are required for development. Use the committed `package-lock.json` for reproducible dependencies. In Windows PowerShell, use `npm.cmd` / `npx.cmd` if execution policy blocks the `.ps1` wrappers.

```powershell
npm.cmd ci
npm.cmd run dev
npm.cmd run build
npm.cmd test
npm.cmd run check:coverage
npx.cmd playwright install chromium
npm.cmd run test:e2e
python -m pytest -q python_examples/04_modules_data/test_smoke.py
python -m pytest -q python_examples/05_ann_gpu/test_smoke.py
python -m pytest -q python_examples/06_regularization_optuna/test_smoke.py
python -m pytest -q python_examples/07_convolutional_networks/test_smoke.py
python -m pytest -q python_examples/08_transfer_learning/test_smoke.py
python -m pytest -q python_examples/09_recurrent_networks/test_smoke.py
python -m pytest -q python_examples/10_faq_classifier/test_smoke.py
python -m pytest -q python_examples/11_lstm_internals/test_smoke.py
python -m pytest -q python_examples/12_lstm_next_word/test_smoke.py
```

`npm.cmd run dev` starts the local Vite server. `npm.cmd run typecheck`, `npm.cmd test`, and `npm.cmd run check:coverage` run the TypeScript, content/snippet and source-coverage checks. Browser tests open the **built file URL** with network access disabled and check navigation, search, bookmarks, themes, diagrams, copy/download controls, print, reduced motion, mobile behavior, and automated accessibility scans.

## Production build

Run `npm ci` and `npm run build`. TypeScript is checked before Vite builds the application. The output includes `dist/index.html` for GitHub Pages and `dist/PyTorch-Atlas.html` as the matching self-contained offline export. `npm run verify:dist` checks that both files exist, are identical, and contain no runtime asset fetches or external asset URLs.

## Offline HTML

Open `dist/PyTorch-Atlas.html` directly in a modern browser. It embeds the application runtime, styles, diagrams, and lesson content. The GitHub Pages entry point and offline file are copied from the same production build.

## GitHub Pages

The repository-based deployment URL will be added here after the repository is created and the live site is verified. The Vite build uses relative asset paths, which work at the repository subpath and from the standalone file. See [DEPLOYMENT.md](DEPLOYMENT.md).

## CI/CD

`.github/workflows/ci.yml` checks pull requests, pushes, and manual runs: npm install, TypeScript, content/snippet/coverage tests, production artifacts and browser accessibility/end-to-end tests. `.github/workflows/deploy.yml` repeats validation and deploys only the repository's default branch through GitHub's Pages artifact and deployment actions. Fork pull requests cannot publish a production deployment.

```powershell
python -m pip install -r python_examples/requirements.txt
python scripts/smoke-python.py
python -m pytest -q python_examples/02_foundations_autograd/test_smoke.py
python -m pytest -q python_examples/03_manual_training/test_smoke.py
python -m pytest -q python_examples/04_modules_data/test_smoke.py
python -m pytest -q python_examples/05_ann_gpu/test_smoke.py
python -m pytest -q python_examples/06_regularization_optuna/test_smoke.py
python -m pytest -q python_examples/07_convolutional_networks/test_smoke.py
python -m pytest -q python_examples/08_transfer_learning/test_smoke.py
python -m pytest -q python_examples/09_recurrent_networks/test_smoke.py
python -m pytest -q python_examples/10_faq_classifier/test_smoke.py
```

The phase folders mirror the lesson snippets from foundations through the FAQ classifier. Pytest suites execute CPU-safe shape, arithmetic, parameter, preprocessing, sequence-gradient, variable-length batching, checkpoint and synthetic-training assertions. The full VGG16 lab is gated because it downloads ImageNet weights and Fashion-MNIST and may require substantial memory/compute.

The final next-word project has explicit CLI entry points and a tiny inline corpus:

```powershell
python python_examples/12_lstm_next_word/lstm_next_word.py train
python python_examples/12_lstm_next_word/lstm_next_word.py predict --seed-text "python libraries"
```

It saves model weights with the vocabulary and configuration. `--corpus-file` accepts an explicitly selected local UTF-8 corpus; see [the project README](python_examples/12_lstm_next_word/README.md). The tiny corpus demonstrates mechanics and does not measure language quality.

## Source and implementation map

- `AGENTS.md`: contributor contract and phase boundaries.
- `docs/source-coverage.json`: final reviewed mappings for PDF pp. 1–211, plus 16 source blocks, 124 source-derived topics, 45 supplemental records and nine source-correction entries.
- `docs/source-audit.md`: source issues, exact boundary corrections, and repairs for future phases.
- `docs/source-pages/`: complete page-ordered extraction for local auditing. Raw page text and the excerpt index are excluded from Git and **never bundled** in the reader.
- `src/content/`: typed lessons, module manifest, snippet registry, revision questions, and source references.
- `src/app/`, `src/components/`: reader shell and reusable learning/navigation controls.
- `src/diagrams/`: fifty-six authored SVG concepts, including convolution controls, transfer stages, sequence shapes, RNN unrolling, BPTT, FAQ flows, LSTM explorers and the next-word prefix, padding, shape, sampling and append steppers.
- `src/design/`: theme tokens, responsive reader layout, focus, reduced-motion and print styles.
- `src/utils/`: persistence, clipboard, and download utilities.
- `python_examples/01_previews/` through `12_lstm_next_word/`: mirrored examples and a consolidated CPU CLI project for autograd, training, data, ANN/device, regularization/search, CNNs, transfer learning, RNNs, FAQ classification and LSTM language modeling.
- `tests/`: content and Playwright tests.
- `docs/verification-report.md`: phase-by-phase checks and the final release audit.
- `docs/screenshots/`: browser captures for visual review.

The canonical Python text lives in the typed snippet registry. After modifying it, run `npm.cmd run snippets:sync`; tests verify the mirrors match without changing them. Keep mathematical assertions in the scripts.

To reproduce extraction from the unchanged root PDF, run `npm.cmd run audit:pdf`. `node scripts/read-pages.mjs 1 5` reads a range without the duplicate text layer. `node scripts/render-source.mjs` renders selected source pages and browser-generated print samples into `tmp/pdf-review/`; run browser tests first to create print samples. This rendering utility uses PDF.js's optional canvas dependency and is not part of the distributed reader.

## Release status

All 94 lessons and 236 lesson code slots are implemented. The source coverage index maps all 211 PDF pages to reviewed topics, records supplemental additions and lists nine groups of source corrections. The original reference PDF and extracted source-page text are excluded from Git. Educational coverage is attributed to the source by page range in the reader; the project does not redistribute the reference PDF.

## Source attribution

Lessons identify source-derived topics with PDF page ranges and label supplemental engineering guidance separately. The reference is a third-party educational PyTorch guide; the original PDF remains local and is excluded from publication. See `docs/source-audit.md` and `docs/source-coverage.json` for attribution and coverage records.
