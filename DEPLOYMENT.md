# Deployment guide

## First-time setup

1. Push the project to the `PyTorch-Atlas` GitHub repository. The original `PYTORCH Guide - Google Gemini.pdf` and extracted `docs/source-pages/` are ignored and must remain untracked.
2. In **Repository → Settings → Pages → Build and deployment → Source**, select **GitHub Actions**.
3. Ensure Actions are enabled for the repository. No secrets, personal access token, or external hosting account are required by the workflows.
4. Push to the repository's default branch or run **Deploy PyTorch Atlas to GitHub Pages** with that branch selected.

## What the workflows do

`ci.yml` runs on pull requests, pushes, and manual dispatch. It installs the lockfile dependencies, checks TypeScript, lesson/snippet coverage, builds both HTML outputs, verifies they are self-contained, and runs Playwright browser/accessibility checks.

`deploy.yml` runs for pushes and manual dispatch, but its build and deploy jobs proceed only when the selected ref is the repository's default branch. It repeats validation before uploading `dist/` with `upload-pages-artifact` and publishing with `deploy-pages` to the `github-pages` environment. Pull request code never reaches the production deployment job.

The deploy workflow sets `contents: read`, `pages: write`, and `id-token: write` only where needed. Deployment runs are serialized and are not cancelled while publishing.

## Verify a deployment

1. Open the Pages URL shown in the successful workflow's `github-pages` environment.
2. Check an Autograd lesson and an ANN, CNN, RNN, and LSTM lesson; navigate by sidebar and in-lesson links.
3. Try `/` search, a diagram control, a code Copy control, theme switching, bookmarks and resume behavior.
4. Test the mobile navigation in a narrow viewport and reload a bookmarked lesson.
5. Download `PyTorch-Atlas.html` from the workflow artifact or regenerate it locally with `npm ci` then `npm run build`.
6. Open the standalone HTML with networking disabled. It should render the same reader as the hosted entry point.
7. In Actions, confirm the CI workflow and the Pages deployment workflow both completed successfully for the deployed commit.

## Troubleshooting

- **Pages deployment says Pages is disabled or the publishing source is wrong:** select **Repository → Settings → Pages → Build and deployment → Source → GitHub Actions**, then rerun deployment.
- **Deploy job is skipped:** it only publishes from the repository's current default branch. Select that branch for manual dispatch or merge the validated change to it.
- **The site loads without styles or script:** retain Vite's `base: "./"` setting and check `npm run verify:dist`; do not change the site to root-only asset paths.
- **Offline copy does not work:** open the actual `dist/PyTorch-Atlas.html`, not `dist/index.html` from a different build. Regenerate both with the same `npm run build` command.
- **A lesson preference is not remembered:** browser storage may be blocked for file URLs or private browsing. The reader falls back to in-memory state for the current session.
- **Browser tests fail because Chromium is absent:** run `npx playwright install --with-deps chromium` before `npm run test:e2e`.
