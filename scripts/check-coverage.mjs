import fs from "node:fs";
import assert from "node:assert/strict";
const data = JSON.parse(fs.readFileSync("docs/source-coverage.json", "utf8"));
assert.equal(data.phase, 12, "final coverage manifest should describe Phase 12");
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
for (let page = 1; page <= 211; page++) {
  const text = fs.readFileSync(
    `docs/source-pages/${String(page).padStart(3, "0")}.txt`,
    "utf8",
  );
  assert(text.length > 100, `Missing page ${page}`);
  assert(
    data.blocks.some((b) => b.topics.some((t) => page >= t.start && page <= t.end)),
    `No topic mapping on page ${page}`,
  );
}
console.log(
  `Coverage passed: 211 pages, 16 blocks, ${topics} reviewed topics, ${data.supplements.length} reviewed supplements, ${data.corrections.length} correction groups.`,
);
