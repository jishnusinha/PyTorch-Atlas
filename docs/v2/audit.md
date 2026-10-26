# V2 F14 reader audit

Baseline: `main` at `d6de5151d9c016ed6f8146073b402175b33b5055`; fetched `origin/main` before branching. Implementation branch: `feat/v2-f14-reader-repairs`.

## Findings

| Status | Evidence | F14 action |
|---|---|---|
| Verified bug | `src/app/App.tsx` rendered `lesson.sections[0]` and `[1]` only; the navigation list was a fixed array. | Render every authored section in order. Keep the original `understand` and `mechanics` anchors, namespace later anchors by lesson ID, and build the TOC from those rendered anchors plus the fixed instructional blocks. |
| Verified bug | `src/components/Search.tsx` claimed only three previews were indexed, and its no-results copy excluded pending chapters. | Describe the live lesson and Python-example index; remove the obsolete empty-state claim. Add additional snippet titles, descriptions, dependencies, shapes, code, expected output and line explanations to search. |
| Verified bug | The quick-reference page and footer in `src/app/App.tsx` described future chapters and displayed `PHASE 01`; `src/content/lessons.ts` and `src/content/phase03.ts` had stale planned/arriving statements. | Replace reader-facing release copy with current navigation guidance and describe already available lessons directly. Remove historical phase wording from the shipped source-index UI. |
| Improvement | `src/components/Learning.tsx` used the generic label “Source-derived concept.” The manifest separates source-derived topics, supplemental additions and correction groups. | Label lesson content “PDF-derived” or “Supplemental engineering”; label entries in the manifest’s correction list “Corrected source example.” No correction is assigned to an individual lesson without supporting lesson metadata. |
| Verified gap | `tests/content.test.ts` checked topic lesson references but did not validate all lesson section fields, lesson source spans or source block bounds against the registry. | Add registry and page-range assertions, plus a rendered four-section fixture and additional-snippet-only search case. |

## Source decision

F14 repairs reader behavior and audits existing mappings; it authors no new curriculum concepts and changes no PDF mapping. The current coverage manifest remains the source of truth for 211 page mappings, 16 source blocks, supplements and correction groups. Correction labels appear on correction records only; lesson provenance is not inferred from page overlap. The local PDF and extracted page files remain ignored and are not copied into the site.

## Stable navigation and preservation

Existing lesson IDs, `#understand`, `#mechanics`, other built-in section IDs, the `#reference` route, bookmarks, progress keys and source-page numbers remain stable. Extra section IDs use `lesson-{lessonId}-{sectionId}`. Current tests cover local saved state, hash navigation, source coverage and the built offline file.
