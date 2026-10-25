import fs from "node:fs/promises";

const files = ["dist/index.html", "dist/PyTorch-Atlas.html"];
const [entry, offline] = await Promise.all(files.map((file) => fs.readFile(file, "utf8")));
if (!entry.includes('<div id="root"></div>')) throw Error("Pages entry has no React root");
for (const [label, html] of [["Pages entry", entry], ["Offline export", offline]]) {
  if (!html.includes("PyTorch Atlas")) throw Error(`${label} is missing the application title`);
  if (/<(?:script|link|img)[^>]+(?:src|href)=["'](?:https?:|\/assets)/i.test(html))
    throw Error(`${label} references a non-relative external asset`);
  if (/\bfetch\(/.test(html)) throw Error(`${label} contains a runtime fetch call`);
  if (!html.includes("type=\"module\"") && !html.includes("type='module'"))
    throw Error(`${label} has no bundled application module`);
}
if (entry !== offline) throw Error("Pages entry and offline HTML must be identical builds");
console.log("Distribution verified: Pages entry and standalone offline HTML are identical and self-contained.");
