import fs from "node:fs";
const start = Number(process.argv[2]),
  end = Number(process.argv[3]);
for (let page = start; page <= end; page++) {
  let text = fs.readFileSync(
    `docs/source-pages/${String(page).padStart(3, "0")}.txt`,
    "utf8",
  );
  const repeat = text.indexOf(text.slice(0, 100), 100);
  if (repeat > 0) text = text.slice(0, repeat);
  console.log(`\n=== PDF page ${page} ===\n${text}`);
}
