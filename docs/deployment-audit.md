# Phase 13 deployment audit

Audited 26 October 2026 before deployment changes. This report distinguishes local verification from GitHub-hosted deployment, which has not yet been possible.

## Initial audit

- The project is a Vite 7, React 19, TypeScript 5.9 reader with `vite-plugin-singlefile`, a lockfile, typed content, authored SVG diagrams, and an existing Playwright suite. All twelve phases are present (94 lessons).
- Navigation uses lesson IDs in the URL hash and does not use React Router, so static hosting does not need server rewrites.
- Vite already uses `base: "./"`, suitable for the repository subpath and the offline file. The previous packaging step renamed the only built entry point, so the Pages build now preserves `dist/index.html` and copies that same self-contained file to `dist/PyTorch-Atlas.html`.
- The initial Git repository had no commits, remotes, or tracked files. The connected GitHub account is `jishnusinha`; its accessible repository list initially contained no `PyTorch-Atlas` repository. The public repository was subsequently created, and the prepared commits were pushed through the authenticated GitHub connection.
- No ESLint configuration or lint script exists. The source PDF, extracted PDF page text, generated `dist/`, dependencies, local environments, and caches are excluded from Git.

## Changes prepared

- Added least-privilege CI and GitHub Pages deployment workflows. Both validate source and production artifacts; deployment is restricted to the repository's default branch and uses the `github-pages` environment.
- Added a Pages `index.html` alongside the standalone offline HTML, relative Vite assets, title/description/Open Graph/Twitter metadata, and an embedded SVG favicon.
- Added README deployment/build/source attribution sections, `DEPLOYMENT.md`, and `npm run verify:dist`.
- Committed the publication candidate on `main`: `8f3f92193b012b499c67632c2c8f433c4bac603a` (`Prepare PyTorch Atlas for GitHub Pages deployment`), followed by audit commit `31889ea71f3ff84a894948861980acbb20d99282`.

## Local checks

- TypeScript: pass (`npm run typecheck`).
- Content and source tests: pass (18/18); coverage pass (211 pages, 16 blocks, 124 reviewed topics, 45 supplements, nine correction groups).
- Production build: pass. `dist/index.html` and `dist/PyTorch-Atlas.html` are identical, self-contained files (875,659 bytes; 267.81 KB gzip).
- Distribution verification: pass; both entries boot independently using offline `file://` Playwright runs with no remote requests or page errors.
- Browser suite: 27/27 pass, including all-lesson accessibility scans, navigation, search, storage, copy/download, diagrams, mobile, print and reduced-motion checks. The separate Pages entrypoint smoke test passed 1/1.
- A local `npm ci` attempt stopped with Windows `EPERM` because the already-running Vite server held `esbuild.exe`. Dependencies were restored using `npm install --ignore-scripts`, then typecheck, content tests, build and artifact verification passed. A clean `npm ci` still needs confirmation in the GitHub-hosted runner.
- The initial hosted CI and deployment runs reached the content test, then failed because that test also validates the built offline artifact. The workflows now build and verify artifacts before running content tests; this correction is being pushed and validated.
- No ESLint check is available because ESLint is not configured. The live Pages site remains unverified.

## Remaining release steps

Repository: <https://github.com/jishnusinha/PyTorch-Atlas>. After the workflow correction passes, check **Repository → Settings → Pages → Build and deployment → Source → GitHub Actions** if the first Pages deployment reports that Pages is not enabled. Confirm the `github-pages` workflow result and open its reported URL; only then record a live URL here.
