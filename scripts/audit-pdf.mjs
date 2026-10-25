import fs from "node:fs/promises";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
await fs.mkdir("docs/source-pages", { recursive: true });
const pdf = await getDocument({
  data: new Uint8Array(await fs.readFile("PYTORCH Guide - Google Gemini.pdf")),
  useSystemFonts: true,
  isEvalSupported: false,
}).promise;
const index = [];
for (let n = 1; n <= pdf.numPages; n++) {
  const page = await pdf.getPage(n);
  const content = await page.getTextContent();
  let text = "";
  let y = null;
  for (const item of content.items) {
    if (!("str" in item)) continue;
    if (y !== null && Math.abs(y - item.transform[5]) > 3) text += "\n";
    text += item.str + " ";
    y = item.transform[5];
  }
  await fs.writeFile(`docs/source-pages/${String(n).padStart(3, "0")}.txt`, text);
  index.push({ page: n, characters: text.length, opening: text.slice(0, 180) });
}
await fs.writeFile("docs/pdf-page-index.json", JSON.stringify(index, null, 2));
console.log(`Extracted ${pdf.numPages} complete pages in order.`);
