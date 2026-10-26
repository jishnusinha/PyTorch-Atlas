import fs from "node:fs";
import assert from "node:assert/strict";
const data = JSON.parse(fs.readFileSync("docs/source-coverage.json", "utf8"));
assert.equal(data.phase, 12, "final coverage manifest should describe Phase 12");
const hasLocalPdfAudit =
  fs.existsSync("PYTORCH Guide - Google Gemini.pdf") &&
  fs.existsSync("docs/source-pages");
const covered = new Set();
let topics = 0;
for (const b of data.blocks) {
  assert(b.start >= 1 && b.end <= 211 && b.start <= b.end);
  for (let p = b.start; p <= b.end; p++) covered.add(p);
  for (const t of b.topics) {
    assert(t.start >= 1 && t.end <= 211 && t.start <= t.end);
    assert(/^[a-z0-9-]+$/.test(t.lesson));
    assert(["planned", "draft", "reviewed"].includes(t.status));
    assert.equal(t.status, "reviewed", `${t.lesson} is not reviewed`);
    assert.equal(t.kind, "source-derived");
    topics++;
  }
}
assert(data.supplements.every((item) => item.status === "reviewed"));
assert(data.corrections.length >= 1);
assert.equal(covered.size, 211);
assert.equal(data.blocks.length, 16);
assert.equal(data.buildProjects?.length, 5, "F17 project/source mapping is required");
const projectIds = new Set();
for (const project of data.buildProjects) {
  assert(/^[a-z0-9-]+$/.test(project.id));
  assert(!projectIds.has(project.id), `duplicate project mapping ${project.id}`);
  projectIds.add(project.id);
  assert(["source-derived", "supplemental"].includes(project.kind));
  assert(project.lessons.length > 0 && project.note.length > 40);
  if (project.kind === "source-derived") {
    assert(Array.isArray(project.pages) && project.pages.length === 2);
    const [start, end] = project.pages;
    assert(start >= 1 && end <= 211 && start <= end);
    for (let page = start; page <= end; page++) {
      assert(data.blocks.some((block) => block.topics.some((topic) => topic.start <= page && topic.end >= page)), `No source topic maps project page ${page}`);
    }
  } else {
    assert.deepEqual(project.pages, [], `${project.id} supplemental material must not claim source pages`);
  }
}
for (let page = 1; page <= 211; page++) {
  if (hasLocalPdfAudit) {
    const text = fs.readFileSync(
      `docs/source-pages/${String(page).padStart(3, "0")}.txt`,
      "utf8",
    );
    assert(text.length > 100, `Missing page ${page}`);
  }
  assert(
    data.blocks.some((b) => b.topics.some((t) => page >= t.start && page <= t.end)),
    `No topic mapping on page ${page}`,
  );
}
console.log(
  `Coverage passed: 211 pages, 16 blocks, ${topics} reviewed topics, ${data.supplements.length} reviewed supplements, ${data.corrections.length} correction groups${hasLocalPdfAudit ? "; complete local PDF text checked" : "; source text not required"}.`,
);
