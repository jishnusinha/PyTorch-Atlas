# F16 Field Kit audit and change map

## Inventory

- The reader has 94 stable lessons and 236 snippet placements in `allSnippets`.
- Snippet placements resolve to 135 stable IDs. Repeated IDs point to byte-identical code; there are no distinct IDs with identical code in the current registry.
- The current `#reference` view renders a 94-lesson revision list and a short autograd cheat sheet. Search indexes lesson text only. `copyText`, `downloadPython`, and `readStored` already provide offline-compatible primitives.
- Python mirrors are organized by module. The content tests verify every placement against the canonical mirror, so this phase must preserve those IDs and code bodies.
- The first curated pass revealed two separate snippets with the same “Reusable derivative check” purpose. F16 keeps the numeric derivative comparison and uses the distinct multi-input gradient example in its place.

## Canonicalization rules

1. Select at most one recipe for each stable snippet ID. `allSnippets` is a placement list, not a second recipe registry.
2. Resolve recipe code from the canonical snippet object. Recipe metadata must never rewrite, normalize, append comments to, or otherwise mutate the copied/downloaded `.py` code.
3. Do not add a second recipe when an existing snippet teaches the same operation. A missing topic may be added only as a new stable snippet with a mirrored Python file and a smoke test.
4. Mark code as runnable only when the canonical example is self-contained and its dependencies are stated. A contextual recipe may be copied, but it must not offer `.py` download.
5. Mark optional dataset/weight downloads in the dependency and run notes. No recipe should trigger a download from the browser.
6. `sourceKind` follows the linked lesson when the snippet is that lesson's primary source-derived example; templates, additional snippets, and supplemental lessons remain labeled supplemental.

## F16 change map

- Add typed recipe metadata over selected canonical snippets, grouped into the nine requested categories. The curated inventory contains 94 distinct recipes; omit near-duplicates and keep the resulting category counts explicit in tests and this report.
- Add `FieldKitHome` and `FieldKitRecipe` components. Only the selected recipe's code detail is mounted. Support category, search, tags, level, favorites, copy, safe download, shape contract, output, gotcha, and lesson backlink.
- Route `#field-kit` and `#field-kit/<recipe-id>`; retain `#reference` as an alias. Keep the existing lesson completion/bookmark storage keys untouched and store recipe favorites under a new namespaced key.
- Extend global search with concise recipe metadata and code-behavior text. Recipe results navigate directly to `#field-kit/<recipe-id>`.
- Update the old quick-reference navigation label and the old reference browser test. Add content integrity and browser coverage for recipe count, routing, filters, favorites, copy/download availability, and mobile layout.

## Verification plan

- `npm test`, `npm run typecheck`, and `npm run test:e2e`.
- CPU-safe Python smoke checks for standalone recipes that have not already been checked by the mirrored snippet suite; explicitly report optional dependencies that are skipped.
- `npm run build` and `npm run verify:dist`; inspect the one-file artifact size and offline parity.
- Accessibility checks for labels, keyboard operation, dialog/search focus, and narrow-screen layout.

## Phase handoff inventory

- The selection contains 94 recipes over nine categories. Reuse ratio: 94/94 code bodies resolve to existing canonical snippets; F16 adds no duplicate code and no Python scripts.
- Category inventory: Tensors 8; Autograd 10; nn.Module 8; Dataset/Loader 10; Losses/Metrics 9; Training/Optimizers 14; CNN/Transfer 12; RNN/LSTM/Text 17; Checkpoints/Debugging 6.
- Source labels: 50 source-derived and 44 supplemental recipes, inherited from the linked lesson's source classification. No new PDF pages were added in F16; all 211 pages remain mapped by the existing manifest.
- The sole non-downloadable contextual recipe is `local-csv-loader`, which needs a user-provided CSV path. Optuna entries state their optional dependencies. No Field Kit action downloads data or weights in the browser.
- Browser screenshots are captured from the standalone file build at `docs/screenshots/field-kit-home.png` and `docs/screenshots/field-kit-recipe.png`.
