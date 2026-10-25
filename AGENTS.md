# PyTorch Atlas contributor contract

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


## Phase workflow
Read the prompt pack and existing source before each phase. Stop and report after each phase. Do not alter the source PDF, push, or deploy. Audit all relevant pages, reconstruct incomplete exports, run CPU-safe snippet checks, content tests, browser tests, and the standalone build. Use nn.Module and model(x), correct logits/loss pairs, train/eval contexts, and leakage-free validation. Report actual results and unresolved limitations. Never label planned content as complete.
