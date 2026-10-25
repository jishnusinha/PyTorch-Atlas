import fs from "node:fs/promises";
const entry = "dist/index.html";
const offline = "dist/PyTorch-Atlas.html";
const html = await fs.readFile(entry, "utf8");
if (/<(?:script|link|img)[^>]+(?:src|href)=["'](?:https?:|\/assets)/i.test(html))
  throw Error("External asset in build");
if (/\bfetch\(/.test(html)) throw Error("Runtime fetch in build");
await fs.copyFile(entry, offline);
console.log(`Pages entry: ${entry}; standalone HTML: ${offline} (${Math.round(Buffer.byteLength(html) / 1024)} KB).`);
