# Phase 13 deployment audit

Audited 26 October 2026 before deployment changes. This report distinguishes local verification from GitHub-hosted deployment, which has not yet been possible.

## Initial audit

- The project is a Vite 7, React 19, TypeScript 5.9 reader with `vite-plugin-singlefile`, a lockfile, typed content, authored SVG diagrams, and an existing Playwright suite. All twelve phases are present (94 lessons).
- Navigation uses lesson IDs in the URL hash and does not use React Router, so static hosting does not need server rewrites.
- Vite already uses `base: "./"`, suitable for the repository subpath and the offline file. The previous packaging step renamed the only built entry point, so the Pages build now preserves `dist/index.html` and copies that same self-contained file to `dist/PyTorch-Atlas.html`.
- The initial Git repository had no commits, remotes, or tracked files. The connected GitHub account is `jishnusinha`; its accessible repository list contains no `PyTorch-Atlas` repository. GitHub CLI is not installed, and the connected GitHub tools do not expose repository creation.
- No ESLint configuration or lint script exists. The source PDF, extracted PDF page text, generated `dist/`, dependencies, local environments, and caches are excluded from Git.

## Changes prepared

- Added least-privilege CI and GitHub Pages deployment workflows. Both validate source and production artifacts; deployment is restricted to the repository's default branch and uses the `github-pages` environment.
- Added a Pages `index.html` alongside the standalone offline HTML, relative Vite assets, title/description/Open Graph/Twitter metadata, and an embedded SVG favicon.
- Added README deployment/build/source attribution sections, `DEPLOYMENT.md`, and `npm run verify:dist`.
- Committed the local publication candidate on `main`: `8f3f92193b012b499c67632c2c8f433c4bac603a` (`Prepare PyTorch Atlas for GitHub Pages deployment`). No remote was added and no push occurred.

## Local checks

- TypeScript: pass (`npm run typecheck`).
- Content and source tests: pass (18/18); coverage pass (211 pages, 16 blocks, 124 reviewed topics, 45 supplements, nine correction groups).
- Production build: pass. `dist/index.html` and `dist/PyTorch-Atlas.html` are identical, self-contained files (875,659 bytes; 267.81 KB gzip).
- Distribution verification: pass; both entries boot independently using offline `file://` Playwright runs with no remote requests or page errors.
- Browser suite: 27/27 pass, including all-lesson accessibility scans, navigation, search, storage, copy/download, diagrams, mobile, print and reduced-motion checks. The separate Pages entrypoint smoke test passed 1/1.
- A local `npm ci` attempt stopped with Windows `EPERM` because the already-running Vite server held `esbuild.exe`. Dependencies were restored using `npm install --ignore-scripts`, then typecheck, content tests, build and artifact verification passed. A clean `npm ci` still needs confirmation in the GitHub-hosted runner.
- No ESLint check is available because ESLint is not configured. No live Pages URL or Actions run exists yet.

## Remaining release steps

Create the public repository `jishnusinha/PyTorch-Atlas` (public visibility was confirmed), push branch `main`, then set **Repository → Settings → Pages → Build and deployment → Source → GitHub Actions**. The GitHub integration does not provide repository-creation operations and this environment has no GitHub CLI, so the local commit cannot be pushed from this session. After those steps, run and inspect both workflows and verify the live site before publishing its URL.
